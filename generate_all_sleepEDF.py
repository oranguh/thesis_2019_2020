import numpy as np
import torch
import mne
from mne.datasets.sleep_physionet.age import fetch_data
import os
from tqdm import tqdm
import random
import pickle as pkl
from visualizations import plot_classes_distribution


mapping = {'EEG Fpz-Cz': 'eeg',
           'EEG Pz-Oz': 'eeg',
           'EOG horizontal': 'eog',
           'Resp oro-nasal': 'misc',
           'EMG submental': 'misc',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}

annotation_desc_2_event_id = {'Sleep stage W': 1,
                              'Sleep stage 1': 2,
                              'Sleep stage 2': 3,
                              'Sleep stage 3': 4,
                              'Sleep stage 4': 4,
                              'Sleep stage R': 5}

event_id = {'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3/4': 4,
            'Sleep stage R': 5}

time_scale = 30.
sampling_Hz = 100.

folder_to_save = 'data'
datapoint_counter = 0
labels_dict = {}

for test_subject in tqdm(fetch_data(subjects=np.arange(19), recording=[1, 2])):
    print("\n")
    raw_train = mne.io.read_raw_edf(test_subject[0])
    annot_train = mne.read_annotations(test_subject[1])

    raw_train.set_annotations(annot_train, emit_warning=False)
    raw_train.set_channel_types(mapping)

    events_train, _ = mne.events_from_annotations(
        raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    tmax = time_scale - 1. / raw_train.info['sfreq']  # tmax in included

    epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                              event_id=event_id, tmin=0., tmax=tmax, baseline=None, preload=True)

    epochs_train = epochs_train.resample(sampling_Hz, npad='auto')

    x = epochs_train.get_data()

    labels = epochs_train.events[:, 2]
    EEG_pz_z = x[:, 0, :]

    zipped = zip(EEG_pz_z, labels)
    no_wake = np.asarray(list(filter(lambda x: x[1] != 1, list(zipped))))

    event_count = no_wake.shape[0]
    event_size = no_wake[0][0].shape[0]

    # This is ugly but it works so whatever

    no_wake_data = np.vstack(no_wake[:, 0]).reshape([event_count, event_size, 1])
    no_wake_labels = np.vstack(no_wake[:, 1]).reshape([event_count])

    # print(no_wake_data.shape, "\n", no_wake_labels.size)

    for i, signal in enumerate(no_wake_data):
        signal = torch.from_numpy(signal)
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)
        torch.save(signal, os.path.join(folder_to_save, str(datapoint_counter)) + '.pt')

        labels_dict[str(datapoint_counter)] = no_wake_labels[i] - 2

        datapoint_counter += 1

    # print(len(labels_dict))


data_shuffled = np.arange(len(labels_dict))
random.shuffle(data_shuffled)
split_point = int(np.round(len(data_shuffled)/3))

partition = {'validation': [str(x) for x in data_shuffled[:split_point]],
             'train': [str(x) for x in data_shuffled[split_point:]]}

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


save_obj(partition, os.path.join(folder_to_save, 'partition'))

save_obj(labels_dict, os.path.join(folder_to_save, 'labels'))

# print(partition['train'], '\n\n', labels)
print("{} Datapoint created. {} Training and {} Validation".format(len(labels_dict), len(partition['train']), len(partition['validation'])))
plot_classes_distribution(labels_dict, ['N1', 'N2', 'N3/4', 'REM'])