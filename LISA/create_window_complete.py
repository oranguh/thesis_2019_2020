import numpy as np
import csv
import torch
import os
import random
import pickle as pkl
import scipy.io
from scipy import stats
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import more_itertools
from tqdm import tqdm
import platform
import copy

"""
create whole window dataset and allow it to be loaded into memory.

994 records (rows)
only 1 channel (for sake of simplicity)
whole recording.

Data: list of lists (of varying size)
Annotations: list of lists. annotation should be integer derived from 'mode' of the annotation window.

for the partition we make tuples (i, j) where i is the row index and j is the window index.
"""

if platform.system() == 'Windows':
    folder_patterson = 'E:\\data\\you-snooze-you-win-the-physionet-computing-in-cardiology-challenge-2018-1.0.0\\marco'
    records = "E:\\data\\you-snooze-you-win-the-physionet-computing-in-cardiology-challenge-2018-1.0.0\\training\\RECORDS"

else:
    folder_patterson = '/project/marcoh/you_snooze_you_win/marco/'
    records = '/project/marcoh/you_snooze_you_win/marco/RECORDS'

data_list = []
annotation_list_arousal = []
annotation_list_sleep = []
partition_list = []

with open(records, "r") as f:
    for i, line in tqdm(enumerate(csv.reader(f))):
        individual = line[0].strip("/")
        individual_folder = os.path.join(folder_patterson, individual)
        individual_records_ = os.path.join(individual_folder, individual)

        X_ = torch.load(individual_records_ + '_data.pt')
        Y_ = torch.load(individual_records_ + '_labels.pt')

        # add single channel to list (memory intensive!)
        data_list.append(copy.deepcopy(X_[0, :]))
        del X_
        # make the annotation list using mode on windows. each window is 3000
        arousals = Y_[0, :]
        sleep_stage = Y_[1, :]

        annotation_arousal = []
        annotation_sleep = []
        for index, _ in enumerate(arousals[::3000]):

            start = index*3000
            end = (index+1) * 3000
            if arousals[start: end].size < 3000:
                # print()
                # print(arousals[start: end].size, "not large enough", individual)
                continue
            mode = stats.mode(arousals[start: end])
            annotation_arousal.append(mode[0][0])

            mode = stats.mode(sleep_stage[start: end])
            annotation_sleep.append(mode[0][0])

        annotation_list_arousal.append(copy.deepcopy(annotation_arousal))
        annotation_list_sleep.append(copy.deepcopy(annotation_sleep))

        # make the partition indices
        # 6000 = 30 seconds * 200Hz
        # 3000 = 30 seconds * 100Hz

        partition_list += [(i, j) for j in list(range(int(Y_.shape[1] / 3000)))]

        del Y_
        del annotation_arousal
        del annotation_sleep

    print('\n', len(partition_list), '\n\n')

    random.shuffle(partition_list)

    split_point = int(len(partition_list) / 3)

    partition = {'validation': [x for x in partition_list[:split_point]],
                 'train': [x for x in partition_list[split_point:]]}

    print("Saving files....")
    torch.save(annotation_list_sleep, os.path.join(folder_patterson, 'sleep_IID_windows_FULL.pt'))
    del annotation_list_sleep
    print("saved sleep annotations")
    torch.save(annotation_list_arousal, os.path.join(folder_patterson, 'arousal_IID_windows_FULL.pt'))
    print("saved arousal annotations")
    del annotation_list_arousal

    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)
    save_obj(partition, os.path.join(folder_patterson, 'data_partition_IID_windows_FULL'))
    print("saved partition")
    del partition
    save_obj(data_list, os.path.join(folder_patterson, 'data_IID_windows_FULL'))
    print("saved big data")