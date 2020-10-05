"""Copyright 2020 Philips Research. All rights reserved.

Author: Ulf Grossekathoefer <ulf.grossekathofer@philips.com>

This file contains code to access data from the dry_dry_sham data set
as provided by the smart sleep team for machine learning experiments.
"""


from config import DATA_PATHS
import pathlib
import sys
import os

import scipy.io
import scipy.signal
import numpy as np

import pandas as pd
from pathlib import Path

# Add current working directory to python path so we can import config.py
sys.path.append(os.getcwd())

class DataSet(object):

    def __init__(self):
        pass

    def __size__(self):
        raise NotImplementedError()

    def get_test_index(self):
        raise NotImplementedError()

    def get_training_index(self):
        raise NotImplementedError()


class ACDryDry(DataSet):

    def __init__(self):
        self.path = DATA_PATHS['dry_dry_sham']

        self.data_inventory_file_name = self.path / 'RecInfo_updated.xlsx'
        # , names=['ID', 'RecName', 'Use', 'Condition'])
        data_inventory_df = pd.read_excel(self.data_inventory_file_name)
        # filter recordings that are good for use:
        self.data_inventory_df = data_inventory_df[(
            data_inventory_df.Use == 1) & (data_inventory_df.Condition == 'sham')]

    @property
    def subject_ids(self):
        return list(self.data_inventory_df.ID.unique())

    @property
    def recording_ids(self):
        return list(self.data_inventory_df.RecName.unique())

    def get_EEG(self, recording_id, channel='all', freq=100.0):

        # extract row that contains recording_id from inventory:
        row = self.data_inventory_df[self.data_inventory_df.RecName == recording_id]
        # create full EEG bin filename:
        eeg_bin_filename = self.path / \
            str(row.RecName.values[0]) / \
            '{}_EEG.bin'.format(row.RecName.values[0])
        # print(eeg_bin_filename)
        fid = open(eeg_bin_filename, "r")
        data = np.fromfile(fid, dtype=np.float32)
        signal = {}
        if channel in ['all', 'mastoid']:
            # Every ten values starting at 2nd value
            signal['mastoid'] = data[1::10]
            signal['mastoid'] = scipy.signal.lfilter(
                [1, -1], [1, -0.9870], signal['mastoid'])
            # resample to target freq:
            signal['mastoid'] = scipy.signal.resample(
                signal['mastoid'], int(len(signal['mastoid'])*(freq/250.0)))
        if channel in ['all', 'frontal']:
            signal['frontal'] = data[8::10]
            signal['frontal'] = scipy.signal.lfilter(
                [1, -1], [1, -0.9870], signal['frontal'])
            # resample to target freq:
            signal['frontal'] = scipy.signal.resample(
                signal['frontal'], int(len(signal['frontal'])*(freq/250.0)))

        return signal

    def get_hypnogram_DL(self, recording_id, freq=100.0):

        assert freq == 100, 'Only 100Hz are supported'

        # extract row that contains recording_id from inventory:
        row = self.data_inventory_df[self.data_inventory_df.RecName == recording_id]
        # create full EEG bin filename:
        eeg_bin_filename = self.path / \
            str(row.RecName.values[0]) / \
            '{}_EEG.bin'.format(row.RecName.values[0])
        # print(eeg_bin_filename)
        fid = open(eeg_bin_filename, "r")
        data = np.fromfile(fid, dtype=np.float32)
        dl_stages = data[2::10]
        # for the stages, use simpler resample strategy:
        # repeat each sample (=500Hz) and take every 5th sample (=100Hz)
        dl_stages = np.repeat(np.squeeze(dl_stages), 2)[::5]

    def get_hypnogram(self, recording_id, freq=100):
        # extract row that contains recording_id from inventory:
        row = self.data_inventory_df[self.data_inventory_df.RecName == recording_id]
        # create full EEG bin filename:
        sleep_staging_filename = self.path / \
            str(row.RecName.values[0]) / \
            '{}_manual_staging.csv'.format(row.RecName.values[0])
        # print(sleep_staging_filename)

        df = pd.read_csv(sleep_staging_filename, sep=',')
        hypno = np.array(df.user)
        # initialize all to Unknonn
        hypno_out = np.full_like(hypno, fill_value=8, dtype=int)
        hypno_out[hypno == 11] = int(1)  # WAKE
        hypno_out[hypno == 12] = int(0)  # REM
        hypno_out[hypno == 13] = int(-1)  # N1
        hypno_out[hypno == 14] = int(-2)  # N2
        hypno_out[hypno == 15] = int(-3)  # N3

        # squeeze the excess dimension from the hypnogram and upsample to 100Hz
        hypno_out = np.repeat(np.squeeze(hypno_out), 6*freq)

        return hypno_out

    def get_arousal(self, recording_id, freq=100, length=1):

        # extract row that contains recording_id from inventory:
        row = self.data_inventory_df[self.data_inventory_df.RecName == recording_id]
        # create full EEG bin filename:
        arousals_filename = self.path / \
            str(row.RecName.values[0]) / \
            '{}_arousals.xlsx'.format(row.RecName.values[0])
        # print(arousals_filename)
        arousals = pd.read_excel(arousals_filename)

        out = np.zeros(int(length))

        for i, arousal in arousals.iterrows():
            arousal_start = float(arousal.RelativeTime)
            arousal_end = arousal_start + float(arousal.Duration)
            arousal_start_sample = int(arousal_start*freq)
            arousal_end_sample = int(arousal_end*freq)

            if arousal_start_sample < length and arousal_end_sample < length:
                out[arousal_start_sample:arousal_end_sample] = 1
            elif arousal_start_sample < length:
                out[arousal_start_sample:length] = 1

        return out


if __name__ == '__main__':
    p = ACDryDry()
    print(p.subject_ids)
    print(p.recording_ids)
    eeg_100 = p.get_EEG('PP01_PSG2_20181129', freq=100.0, channel='frontal')
    # print(eeg_100)
    # print(eeg_100['frontal'].shape)
    arousal = p.get_arousal('PP01_PSG2_20181129',
                            length=eeg_100['frontal'].shape[0], freq=100)
    # print(arousal.shape)
    slst = p.get_hypnogram('PP01_PSG2_20181129', freq=100)

    print(eeg_100['frontal'].shape, arousal.shape, slst.shape)
    print(asas)

    if 1:
        eeg_100 = p.get_EEG('PP01_PSG2_20181129')
        for key in eeg_100.keys():
            print('100 ', key, eeg_100[key].shape)

        eeg_150 = p.get_EEG('PP01_PSG2_20181129', freq=150.0)
        for key in eeg_150.keys():
            print('150 ', key, eeg_150[key].shape)

        eeg_fron = p.get_EEG('PP01_PSG2_20181129', channel='frontal')
        for key in eeg_fron.keys():
            print('fron', key, eeg_fron[key].shape)

    if 1:
        slst = p.get_hypnogram('PP01_PSG2_20181129', freq=100)
        print('slst 100 ', '    ', slst.shape)

        slst = p.get_hypnogram('PP01_PSG2_20181129', freq=150)
        print('slst 150 ', '    ', slst.shape)

    if 1:
        eeg_fron = p.get_EEG('PP01_PSG2_20181129', channel='frontal')
        for key in eeg_fron.keys():
            print('fron', key, eeg_fron[key].shape)

        arousal = p.get_arousal('PP01_PSG2_20181129',
                                length=eeg_fron[key].shape[0], freq=100)
        print('arou 100 ', key, arousal.shape)

        eeg_fron = p.get_EEG('PP01_PSG2_20181129', channel='frontal', freq=150)
        for key in eeg_fron.keys():
            print('fron', key, eeg_fron[key].shape)
        arousal = p.get_arousal('PP01_PSG2_20181129',
                                length=eeg_fron[key].shape[0], freq=150)
        print('arou 150 ', key, arousal.shape)
