import numpy as np
import torch
import os
from tqdm import tqdm
import random
import pickle as pkl
import scipy.io
import h5py

folder_path = 'H:\data\you-snooze-you-win-the-physionet-computing-in-cardiology-challenge-2018-1.0.0\\training\\'

folder_to_save = './'

datapoint_counter = 0

data_array = np.asarray([])
counter = 0

overwrite = False

for root, dirs, files in tqdm(os.walk(folder_path), total=len(os.listdir(folder_path))):

    if counter == 0:
        counter += 1
        continue

    if counter == 505:
        break

    head_tail = os.path.split(root)
    file = os.path.join(root, head_tail[-1])

    if os.path.isfile(file + "_labels.pt") and not overwrite:
        print(f"{head_tail[-1]} exists, skipping saving")
        data_array = np.append(data_array, head_tail[-1])
        counter += 1
        continue

    X = scipy.io.loadmat(file + ".mat")
    X = X['val']

    # creating the Y matrix is trickier, first pre-allocate
    Y = np.zeros((7, X.shape[-1]), dtype='int32')

    # Then open the annotation file using h5py
    with h5py.File(file + "-arousal.mat", 'r') as f:
        # input arousal labels into first row
        temp_array_ = np.zeros(f['data']['arousals'].shape, dtype='int32')
        f['data']['arousals'].read_direct(temp_array_)
        Y[0] = np.squeeze(temp_array_)

        #        sleep stage labels go into the rest
        for i, key in enumerate(f['data']['sleep_stages'].keys()):
            temp_array_ = np.zeros(f['data']['sleep_stages'][key].shape, dtype='int32')
            f['data']['sleep_stages'][key].read_direct(temp_array_)
            Y[i + 1] = np.squeeze(temp_array_)

    # # # # TODO Pre-processing # # # #
    # Mean and RMS?
    # Some kind of anti aliasing filter?
    # moving window?

    torch.save(X, file + '_data.pt')
    torch.save(Y, file + '_labels.pt')

    print(f"Saved {head_tail[-1]}")

    counter += 1
    data_array = np.append(data_array, head_tail[-1])


print('\n', data_array, '\n\n')

random.shuffle(data_array)
split_point = int(np.round(len(data_array) / 3))

partition = {'validation': [str(x) for x in data_array[:split_point]],
             'train': [str(x) for x in data_array[split_point:]]}


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


save_obj(partition, os.path.join(folder_to_save, 'data_partition'))

print('Training: ',partition['train'], '\nValidation: ',partition['validation'])
