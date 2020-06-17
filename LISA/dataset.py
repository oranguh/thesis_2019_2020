import torch
from torch.utils import data
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


def load_obj(name):
    with open(name, 'rb') as f:
        return pkl.load(f)


class Dataset_full(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, folder, downsample_ratio=2, pre_allocation=3422800):
        'Initialization'
        self.list_IDs = list_IDs
        self.folder = folder
        self.downsample_ratio = downsample_ratio
        self.pre_allocation = pre_allocation

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        folder_ = os.path.join(self.folder, ID)

        # 6609000
        # try:
        X_ = torch.load(os.path.join(folder_, ID + '_data.pt'))
        Y_ = torch.load(os.path.join(folder_, ID + '_labels.pt'))
        # except:
        #     print(ID)
        #     ID = ID + "ERRORRRR"
        #     return ID, torch.zeros((13, 6845600)).double(), torch.zeros((1, 6845600)).long(), torch.zeros((1, 6845600)).long()

        X = np.zeros((X_.shape[0], self.pre_allocation))
        Y = np.full((Y_.shape[0], self.pre_allocation), -1)

        X[:X_.shape[0], :X_.shape[1]] = X_
        Y[:Y_.shape[0], :Y_.shape[1]] = Y_

        y_arousal = Y[0, :]
        # original: -1=unscored; 0=not_arousal; 1=arousal
        y_arousal += 1
        # new: 0=unscored; 1=not_arousal; 2=arousal

        # sleep stage labels ['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake']
        # turn the padding into undefined
        y_sleep = Y[1, :]
        y_sleep[y_sleep < 0] = 4

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
            to_down = to_down[::2]
            to_down = to_down[::5]
            to_down = to_down[::5]
            return to_down

        y_arousal = downsampler(y_arousal)
        y_sleep = downsampler(y_sleep)

        return ID, X, y_arousal, y_sleep


class Dataset_IID_window(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, partition, folder):
        'Initialization, partition should be data_partition_IID_windows_FULL'
        self.list_IDs = partition
        self.folder = folder

        # I have to use pickle because torch.save has memory issues with 10GB saving
        self.recording_data = load_obj(os.path.join(folder, "data_IID_windows_FULL.pkl"))
        self.sleep_annotation = torch.load(os.path.join(folder, "sleep_IID_windows_FULL.pt"))
        self.arousal_annotation = torch.load(os.path.join(folder, "arousal_IID_windows_FULL.pt"))
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
        X = np.expand_dims(X, axis=0)

        return ID, X, y_arousal, y_sleep


class Dataset_full_SHHS(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, folder):
        'Initialization'
        self.list_IDs = list_IDs
        self.folder = folder

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        folder_ = os.path.join(self.folder, ID)

        X_ = np.load(os.path.join(folder_, ID + '_data.npy'))
        Y_ = np.load(os.path.join(folder_, ID + '_labels.npy'))

        if X_.shape[0] != 3:
            print("Size of EEG not corrext")
            print(ID)
            print(X_.shape)

        if Y_.shape[0] != 2:
            print("Size of annotation not correct")
            print(ID)
            print(Y_.shape)

        X = np.full((X_.shape[0], 3597000), 0)
        Y = np.full((Y_.shape[0], 3597000), 4)

        X[:X_.shape[0], :X_.shape[1]] = X_
        Y[:Y_.shape[0], :Y_.shape[1]] = Y_

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

        # turn the padding into undefined
        # y_sleep = Y[1, :]
        # y_sleep[y_sleep < 0] = 4

        # categories_ = [1, 2, 3, 4, 5, 6]
        # y_sleep = np.multiply(Y[1:, :].transpose(), categories_).transpose().sum(axis=0)
        # y_sleep[y_sleep < 0] = 0

        # Downsample from 100Hz to 50Hz
        X = X[:, ::2]

        # Downsample annotations in similar way as model from 100Hz to 1Hz
        def downsampler(to_down):
            # initial 100 to 50Hz
            to_down = to_down[::2]
            # from 50 Hz to 1 Hz
            to_down = to_down[::2]
            to_down = to_down[::5]
            to_down = to_down[::5]
            return to_down

        y_arousal = downsampler(y_arousal)
        y_sleep = downsampler(y_sleep)

        return ID, X, y_arousal, y_sleep
