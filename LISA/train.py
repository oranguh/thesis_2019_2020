import torch
import numpy as np
from torch.utils import data
from tools import accuracy, CustomFormatter
from dataset import Dataset
# from visualizations import plot_confusion_matrix, plot_classes_distribution
import matplotlib.pyplot as plt
from patterson_model import Howe_Patterson
import torch.nn as nn
import torch.optim as optim
import os
import pickle as pkl
from sklearn import metrics
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def load_obj(name):
    with open(name, 'rb') as f:
        return pkl.load(f)


# create logger with 'spam_application'
logger = logging.getLogger("My_app")
logger.setLevel(logging.INFO)

writer = SummaryWriter(comment=str(datetime.now().strftime("%Y%m%d-%H%M%S")))

# create console handler with a higher log level
# ch = logging.StreamHandler()
# ch.setLevel(logging.INFO)
#
# ch.setFormatter(CustomFormatter())
#
# if not logger.handlers:
#     logger.addHandler(ch)


# load metadata stuff
arousal_annotation = ["not_scored", "not_arousal", "Arousal"]
sleep_stages = ['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake']

data_folder = '/project/marcoh/you_snooze_you_win/marco/'
partition = load_obj('/project/marcoh/you_snooze_you_win/marco/data_partition.pkl')

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# device = 'cpu'
# cudnn.benchmark = True

# Parameters
params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 2}
max_epochs = 50
# default 12
channels_to_use = 6

# Generators

training_set = Dataset(partition['train'], data_folder)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], data_folder)
validation_generator = data.DataLoader(validation_set, **params)

dataloaders = {"train": training_generator, "val": validation_generator}

# models

model = Howe_Patterson(channels_to_use=channels_to_use)
model.to(device)

# ignore classes unscored and undefined
# Set weights for class arousal vs not arousal

criterion = nn.CrossEntropyLoss(ignore_index=0)
criterion_sleep = nn.CrossEntropyLoss(ignore_index=0)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print(torch.cuda.get_device_name(device=device))
# Loop over epochs
for epoch in range(max_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        # Training
        true_array = np.empty(0)
        pred_array = np.empty(0)

        true_array_sleep = np.empty(0)
        pred_array_sleep = np.empty(0)

        running_loss = 0.0
        # MAX_LEN = 0

        for inputs, annotations_arousal, annotations_sleep in dataloaders[phase]:

            # Transfer to GPU
            inputs = inputs.to(device)
            annotations_arousal = annotations_arousal.to(device)
            annotations_sleep = annotations_sleep.to(device)

            # plt.plot(annotations_arousal.cpu().numpy())
            # plt.savefig(fname="/project/marcoh/code/arousal.png")
            #
            # plt.plot(annotations_sleep.cpu().numpy())
            # plt.savefig(fname="/project/marcoh/code/sleep.png")

            optimizer.zero_grad()
            # Model computations

            # originally only use first 12, ignore cardiogram?
            # ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'Chin1-Chin2', 'ABD', 'CHEST', 'AIRFLOW', 'SaO2', 'ECG']
            inputs = inputs[:, 0:channels_to_use, :]


            # print(inputs.shape, annotations_arousal.shape)

            # only use first 1/5 of data for memory
            if False:
                bat_ = int(inputs.shape[-1]*.2)
                lab_ = int(annotations_arousal.shape[-1]*.2)

                inputs = inputs[:, :, 0:bat_]
                annotations_arousal = annotations_arousal[:, 0:lab_]
            # print(inputs.shape, annotations_arousal.shape)

            # Downsample to 50Hz
            inputs = inputs[:, :, ::4]

            # Downsample annotations in similar way as model
            def downsampler(to_down):

                to_down = to_down[:, ::4]
                to_down = to_down[:, ::2]
                to_down = to_down[:, ::5]
                to_down = to_down[:, ::5]
                return to_down


            annotations_arousal = downsampler(annotations_arousal)
            annotations_sleep = downsampler(annotations_sleep)

            batch_sz_ = inputs.shape[0]
            data_sz = inputs.shape[-1]

            with torch.set_grad_enabled(phase == 'train'):
                # LSTM data should be sequence, batch, data. BUT only after the CNN part! but I will just set batch_first=True
                inputs = inputs.view([batch_sz_, channels_to_use, data_sz]).type(torch.cuda.FloatTensor).to(device)
                annotations_arousal = annotations_arousal.type(torch.cuda.LongTensor).to(device)

                arousal_out, sleep_out = model.forward(inputs)
                # print("ouput", arousal_out.shape)
                # print(y.shape)
                loss_arousal = criterion(arousal_out, annotations_arousal)
                loss_sleep = criterion(sleep_out, annotations_sleep)

                loss = loss_arousal + loss_sleep
                # print(epoch, phase, "Loss ", loss.item())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            true_array = np.append(true_array, annotations_arousal.cpu().numpy().squeeze()).astype(int)
            pred_array = np.append(pred_array, arousal_out.argmax(dim=1).cpu().numpy().squeeze()).astype(int)

            true_array_sleep = np.append(true_array_sleep, annotations_sleep.cpu().numpy().squeeze()).astype(int)
            pred_array_sleep = np.append(pred_array_sleep, sleep_out.argmax(dim=1).cpu().numpy().squeeze()).astype(int)

            running_loss += loss.item() * batch_sz_

            # print("acc", accuracy(arousal_out.argmax(dim=1).cpu().numpy().squeeze(), annotations_arousal.cpu().numpy().squeeze()))
            # print(np.unique(true_array))
            # print("Max Mem GB  ", torch.cuda.max_memory_allocated(device=device) * 1e-9)
            # print()

        print("Max Mem GB  ", torch.cuda.max_memory_allocated(device=device) * 1e-9)

        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        acc = accuracy(pred_array, true_array)
        acc_sleep = accuracy(pred_array_sleep, true_array_sleep)

        writer.add_scalar('{}: Loss'.format(phase), epoch_loss, global_step=epoch)
        writer.add_scalar('{}: Accuracy arousal'.format(phase), acc, global_step=epoch)
        writer.add_scalar('{}: Accuracy sleep staging'.format(phase), acc_sleep, global_step=epoch)

        report = metrics.classification_report(true_array, pred_array, target_names=arousal_annotation)
        writer.add_text('Report Arousals{}'.format(phase), report + '\n', global_step=epoch)

        sleep_report = metrics.classification_report(true_array_sleep, pred_array_sleep, target_names=sleep_stages)
        writer.add_text('Report Sleep staging{}'.format(phase), sleep_report + '\n', global_step=epoch)

        logger.info("\n{}: epoch: {} Loss {:.3f} Accuracy arousal: {:.3f} Accuracy arousal: {:.3f}\n".format(phase, epoch, epoch_loss, acc, acc_sleep))
        logger.info("\n{}: Arousal Report: \n{}\n".format(phase, report))
        logger.info("\n{}: Sleep Report: \n{}\n".format(phase, sleep_report))


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# print(y_test, "\n", y_pred)

# Plot normalized confusion matrix
fig_confusion = plot_confusion_matrix(true_array, pred_array, classes=arousal_annotation, normalize=True, title='Normalized confusion matrix')
writer.add_figure("Confusion Matrix", fig_confusion)

fig_classes = plot_classes_distribution(labels, classes=arousal_annotation)
writer.add_figure("Class distribution", fig_classes)

writer.close()