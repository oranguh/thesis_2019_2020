from dry_dry_sham import ACDryDry

import torch
from torch.utils import data
import os
import numpy as np
# import matplotlib.pyplot as plt
import pickle as pkl
import random

def load_obj(name):
    with open(name, 'rb') as f:
        return pkl.load(f)


class ConcatDataset(torch.utils.data.Dataset):
    """
    when combining, dataset order matters:
    snooze
    SHHS,
    Philips?,
    HMC?

    is my convention for now

    """
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        # print(self.datasets)
        # print(tuple(d[i] for d in self.datasets))
        return random.choice(tuple(d[i] for d in self.datasets))

    def __len__(self):
        return min(len(d) for d in self.datasets)


class Dataset_full(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, folder, downsample_ratio=2, pre_allocation=3597000, down_sample_annotation=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.folder = folder
        self.downsample_ratio = downsample_ratio
        self.pre_allocation = pre_allocation
        self.down_sample_annotation = down_sample_annotation

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # print(asasa)
        # Load data and get label
        folder_ = os.path.join(self.folder, ID)
        # 6609000
        # try:
        # F3-M2, F4-M1, C3-M2, C4-M1, O1-M2 and O2-M1; one electrooculography (EOG) signal at E1-M2;
        # three electromyography (EMG) signals of chin, abdominal and chest movements; one measure of respiratory
        # airflow; one measure of oxygen saturation (SaO2); one electrocardiogram (ECG)
        # philips (Fpz - M2; Fpz - Fp1)
        X_ = torch.load(os.path.join(folder_, ID + '_data.pt'))
        Y_ = torch.load(os.path.join(folder_, ID + '_labels.pt'))

        if True:
            X_ = X_[0, :]
            X_ = np.expand_dims(X_, axis=0)

        X = np.zeros((X_.shape[0], self.pre_allocation)).astype(float)
        Y = np.full((Y_.shape[0], self.pre_allocation), -1)

        X[:X_.shape[0], :X_.shape[1]] = X_
        Y[:Y_.shape[0], :Y_.shape[1]] = Y_
        del X_
        del Y_

        # X = X[random.randint(0, 3), :]
        # X = np.expand_dims(X, axis=0)

        y_arousal = Y[0, :]
        # original: -1=unscored; 0=not_arousal; 1=arousal
        y_arousal += 1
        # new: 0=unscored; 1=not_arousal; 2=arousal

        # sleep stage labels ['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake']
        # turn the padding into undefined
        y_sleep = Y[1, :]
        y_sleep[y_sleep < 0] = 4

        del Y
        # categories_ = [1, 2, 3, 4, 5, 6]
        # y_sleep = np.multiply(Y[1:, :].transpose(), categories_).transpose().sum(axis=0)
        # y_sleep[y_sleep < 0] = 0

        # np.save("/project/marcoh/code/bsleep", y_sleep)
        # np.save("/project/marcoh/code/brousal", y_arousal)
        # print(y_arousal.shape, y_sleep.shape)

        # Downsample from 100Hz to 50Hz
        X = X[:, ::self.downsample_ratio]

        # Downsample annotations in similar way as model from 100Hz to 1Hz
        def downsampler(to_down):
            # initial 100 to 50Hz
            to_down = to_down[::self.downsample_ratio]
            # from 50 Hz to 1 Hz

            if self.down_sample_annotation:
                to_down = to_down[::2]
                to_down = to_down[::5]
                to_down = to_down[::5]
            return to_down

        y_arousal = downsampler(y_arousal)
        y_sleep = downsampler(y_sleep)
        # print(X.shape, y_arousal.shape)
        # print(asas)
        return ID, X, y_arousal, y_sleep


class Dataset_IID_window(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, partition, folder):
        'Initialization, partition should be: data_partition_IID_windows'
        self.list_IDs = partition
        self.folder = folder

        # I have to use pickle because torch.save has memory issues with 10GB saving
        self.recording_data = load_obj(os.path.join(folder, "data_IID_windows.pkl"))
        self.sleep_annotation = torch.load(os.path.join(folder, "sleep_IID_windows.pt"))
        self.arousal_annotation = torch.load(os.path.join(folder, "arousal_IID_windows.pt"))
        self.list_IDs = partition

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample, will return tuple (i, j) denoting row and index
        ID = self.list_IDs[index]
        row = ID[0]
        index = ID[1]
        y_arousal = self.arousal_annotation[row][index]
        # add 1 to remove negative annotations

        y_arousal += 1
        y_sleep = self.sleep_annotation[row][index]
        # set negative values to unknown
        if y_sleep < 0:
            y_sleep = 4

        # get the window from the index, 3000 is window size
        # sampling frequency
        HZ = 100
        # seconds
        WINDOW = 30
        start = index * HZ * WINDOW
        end = (index + 1) * HZ * WINDOW
        X = self.recording_data[row][start: end]

        # need to expand dims to simulate "channels"
        X = np.expand_dims(X, axis=0).astype(float)
        # print(ID, type(X), type(y_arousal), type(y_sleep))
        # print(asas)
        return ID, X, y_arousal.astype(float), y_sleep


class Dataset_full_SHHS(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, folder, downsample_ratio=2, pre_allocation=3597000, down_sample_annotation=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.folder = folder
        self.downsample_ratio = downsample_ratio
        self.pre_allocation = pre_allocation
        self.down_sample_annotation = down_sample_annotation

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # print(ID)
        # Load data and get label
        folder_ = os.path.join(self.folder, ID)

        X_ = np.load(os.path.join(folder_, ID + '_data.npy'))
        Y_ = np.load(os.path.join(folder_, ID + '_labels.npy'))

        if True:
            X_ = X_[0, :]
            X_ = np.expand_dims(X_, axis=0)

        # X_ = torch.load(os.path.join(folder_, ID + '_data.pt'))
        # Y_ = torch.load(os.path.join(folder_, ID + '_labels.pt'))

        X = np.full((X_.shape[0], self.pre_allocation), 0).astype(float)
        Y = np.full((Y_.shape[0], self.pre_allocation), 4)

        #     EEG (sec): 	C3 	A2
        #     EEG:    C4 	A1

        X[:X_.shape[0], :X_.shape[1]] = X_
        Y[:Y_.shape[0], :Y_.shape[1]] = Y_
        del X_
        del Y_
        # X = X[random.randint(0, 1), :]
        # X = np.expand_dims(X, axis=0)

        y_arousal = Y[-2, :]
        y_arousal[y_arousal == 4] = -1
        # original: -1=unscored; 0=not_arousal; 1=arousal
        y_arousal += 1
        # new: 0=unscored; 1=not_arousal; 2=arousal

        # "Wake"= 1 "REM sleep"= 0 "Stage 1 sleep" = -1 "Stage 2 sleep"= -2 "Stage 3 sleep"= -3  "Stage 4 sleep"= -4
        # Combine sleep stage 4 and 3, add undefined
        # model sleep stage labels ['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake']
        # turn into [0, 1, 2, 3, 4, 5]
        # ORDER MATTERS

        dicto = {0: 3,
                 1: 5,
                 -1: 0,
                 -2: 1,
                 -3: 2,
                 -4: 2,
                 4: 4}

        y_sleep = np.asarray(list(map(dicto.get, Y[-1, :])))

        del Y
        # turn the padding into undefined
        # y_sleep = Y[1, :]
        # y_sleep[y_sleep < 0] = 4

        # categories_ = [1, 2, 3, 4, 5, 6]
        # y_sleep = np.multiply(Y[1:, :].transpose(), categories_).transpose().sum(axis=0)
        # y_sleep[y_sleep < 0] = 0

        # Downsample from 100Hz to 50Hz
        X = X[:, ::self.downsample_ratio]

        # Downsample annotations in similar way as model from 100Hz to 1Hz
        def downsampler(to_down):
            # initial 100 to 50Hz
            to_down = to_down[::self.downsample_ratio]
            # from 50 Hz to 1 Hz
            if self.down_sample_annotation:
                to_down = to_down[::2]
                to_down = to_down[::5]
                to_down = to_down[::5]
            return to_down

        y_arousal = downsampler(y_arousal)
        y_sleep = downsampler(y_sleep)

        return ID, X, y_arousal, y_sleep


class Dataset_IID_window_SHHS(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, partition, folder):
        'Initialization, partition should be data_partition_IID_windows_FULL'
        self.list_IDs = partition
        self.folder = folder

        self.recording_data = load_obj(os.path.join(folder, "data_IID_windows.pkl"))
        self.sleep_annotation = torch.load(os.path.join(folder, "sleep_IID_windows.pt"))
        self.arousal_annotation = torch.load(os.path.join(folder, "arousal_IID_windows.pt"))
        self.list_IDs = partition

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample, will return tuple (i, j) denoting row and index
        ID = self.list_IDs[index]
        row = ID[0]
        index = ID[1]

        y_arousal = self.arousal_annotation[row][index]
        # add 1 to remove negative annotations

        # original: -1=unscored; 0=not_arousal; 1=arousal
        y_arousal += 1
        # new: 0=unscored; 1=not_arousal; 2=arousal

        y_sleep = self.sleep_annotation[row][index]

        dicto = {0: 3,
                 1: 5,
                 -1: 0,
                 -2: 1,
                 -3: 2,
                 -4: 2,
                 4: 4}

        y_sleep = dicto.get(y_sleep)

        # get the window from the index, 3000 is window size
        # sampling frequency
        HZ = 100
        # seconds
        WINDOW = 30
        start = index * HZ * WINDOW
        end = (index + 1) * HZ * WINDOW
        X = self.recording_data[row][start: end]

        # need to expand dims to simulate "channels"
        X = np.expand_dims(X, axis=0).astype(float)
        # print(ID, type(X), type(y_arousal), type(y_sleep))
        # print(asas)
        return ID, X, y_arousal, y_sleep


class Dataset_Philips_full(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, downsample_ratio=2, pre_allocation=3597000, down_sample_annotation=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.downsample_ratio = downsample_ratio
        self.loaders_object = ACDryDry()
        self.pre_allocation = pre_allocation
        self.down_sample_annotation = down_sample_annotation

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        # print(ID)
        # Load data and get label

        X_ = self.loaders_object.get_EEG(ID, channel='frontal', freq=100)
        X_ = X_['frontal']
        y_arousal_ = self.loaders_object.get_arousal('PP01_PSG2_20181129', length=X_.shape[0], freq=100)
        y_sleep_ = self.loaders_object.get_hypnogram('PP01_PSG2_20181129', freq=100)

        # Use only first channel
        if True:
            # X_ = X_[0, :]
            X_ = np.expand_dims(X_, axis=0)

        X = np.full((X_.shape[0], self.pre_allocation), 0).astype(float)
        y_arousal = np.full((X_.shape[0], self.pre_allocation), -1)
        y_sleep = np.full((X_.shape[0], self.pre_allocation), 4)

        print(y_arousal.shape, y_sleep.shape, X.shape, X_.shape)
        X[:X_.shape[0], :X_.shape[1]] = X_
        y_arousal[:y_arousal_.shape[0], :y_arousal_.shape[1]] = y_arousal_
        y_sleep[:y_sleep_.shape[0], :y_sleep_.shape[1]] = y_sleep_

        del X_
        del y_arousal_
        del y_sleep_
        # X = X[random.randint(0, 1), :]
        # X = np.expand_dims(X, axis=0)

        # original: -1=unscored; 0=not_arousal; 1=arousal
        y_arousal += 1
        # new: 0=unscored; 1=not_arousal; 2=arousal

        # "Wake"= 1 "REM sleep"= 0 "Stage 1 sleep" = -1 "Stage 2 sleep"= -2 "Stage 3 sleep"= -3  "Stage 4 sleep"= -4
        # Combine sleep stage 4 and 3, add undefined as 4
        # model sleep stage labels ['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake']
        # turn into [0, 1, 2, 3, 4, 5]
        # ORDER MATTERS

        #         int(1)  # WAKE -> 5
        #         int(0)  # REM -> 3
        #         int(-1)  # N1 -> 0
        #         int(-2)  # N2 -> 1
        #         int(-3)  # N3 -> 2
        #         8 = undefined -> 4

        dicto = {0: 3,
                 1: 5,
                 -1: 0,
                 -2: 1,
                 -3: 2,
                 -4: 2,
                 8: 4,
                 4: 4}

        y_sleep = np.asarray(list(map(dicto.get, y_sleep[-1, :])))

        # turn the padding into undefined
        # y_sleep = Y[1, :]
        # y_sleep[y_sleep < 0] = 4

        # categories_ = [1, 2, 3, 4, 5, 6]
        # y_sleep = np.multiply(Y[1:, :].transpose(), categories_).transpose().sum(axis=0)
        # y_sleep[y_sleep < 0] = 0

        # Downsample from 100Hz to 50Hz
        X = X[:, ::self.downsample_ratio]

        # Downsample annotations in similar way as model from 100Hz to 1Hz
        def downsampler(to_down):
            # initial 100 to 50Hz
            to_down = to_down[::self.downsample_ratio]
            # from 50 Hz to 1 Hz
            if self.down_sample_annotation:
                to_down = to_down[::2]
                to_down = to_down[::5]
                to_down = to_down[::5]
            return to_down

        y_arousal = downsampler(y_arousal)
        y_sleep = downsampler(y_sleep)

        return ID, X, y_arousal, y_sleep
