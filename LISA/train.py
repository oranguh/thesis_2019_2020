import torch
import numpy as np
from torch.utils import data
from tools import accuracy, CustomFormatter, get_dataloader
from dataset import Dataset_full, Dataset_IID_window, Dataset_full_SHHS
# from visualizations import plot_confusion_matrix, plot_classes_distribution
import matplotlib.pyplot as plt
from patterson_model import Howe_Patterson
from ConvNet_IID import ConvNet_IID
from deep_sleep import Deep_Sleep
import torch.nn as nn
import torch.optim as optim
import os
import pickle as pkl
from sklearn import metrics
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.observers import FileStorageObserver
import platform


def initialize_model(model_name, channels_to_use):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    #

    if model_name == "ConvNet_IID":
        """ ConvNet_IID
        """
        model_ft = model = ConvNet_IID(channels_to_use=channels_to_use)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)

        # Window lenght in seconds * Hz of data
        input_size = 30 * 100

    elif model_name == "Howe_Patterson":
        """ Howe_Patterson
        """
        model_ft = Howe_Patterson(channels_to_use=channels_to_use)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

        # largest recording length 3422800 at 100Hz
        input_size = 3422800

    elif model_name == "Deep_Sleep":
        """ Deep_Sleep
        """
        # largest recording length 3422800 at 100Hz
        # For deep sleep we use factors of 2.
        input_size = 2**22

        model_ft = Deep_Sleep(channels_to_use=channels_to_use, input_length=input_size)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def load_obj(name):
    with open(name, 'rb') as f:
        return pkl.load(f)


ex = Experiment("hello")
ex.observers.append(FileStorageObserver('my_runs'))
ex.observers.__dir__()
writer = SummaryWriter(comment=str(datetime.now().strftime("%Y%m%d-%H%M%S")))


@ex.config
def cfg():

    # load metadata stuff
    arousal_annotation = ["not_scored", "not_arousal", "Arousal"]
    weights_arousal = [.0, .1, .9]  # Snooze
    weights_arousal = [.0, .05, .95]  # SHHS
    sleep_stages = ['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake']
    weights_sleep = [.2, .1, .3, .2, .0, .2]  # for Snooze
    weights_sleep = [.3, .05, .3, .3, .0, .05]  # for SHHS

    weights_sleep, weights_arousal = [.2, .2, .2, .2, .0, .2], [.0, .5, .5]

    if platform.system() == 'Windows':
        data_folder = 'E:\\data\\you-snooze-you-win-the-physionet-computing-in-cardiology-challenge-2018-1.0.0\\marco'
        data_folder = 'F:\\shhs\\polysomnography\\shh1_numpy'
        # Parameters for dataloader
        dataloader_params = {'batch_size': 1,
                             'shuffle': True,
                             'num_workers': 0}
    else:
        data_folder = '/project/marcoh/you_snooze_you_win/marco/'
        data_folder = '/project/marcoh/shhs/polysomnography/shh1_numpy/'
        # Parameters for dataloader
        dataloader_params = {'batch_size': 8,
                             'shuffle': True,
                             'num_workers': 8}

    max_epochs = 100
    # full PSG has 12, we use 1
    channels_to_use = 1

    optimizer_params = {"lr": 1e-4,
                        "optim": "Adam"}
    lr = 1e-4

    # models to train... Howe_Patterson, U-Net, CNN, cNN + LSTM, wavenet? Model also determines the way the data is loaded.

    # for conv use 2048 batchsize ["ConvNet_IID", "Howe_Patterson"]
    # model_name = "ConvNet_IID"
    # for howe use batchsize 6 on lisa?
    model_name = "Howe_Patterson"

    # Dataset to use.... also determines path. Snooze, SHHS, Philips, HMC
    # data_name = "snooze"
    data_name = "SHHS"
    # Predicting future labels is a thing I want to focus on.
    # pred_future = True

@ex.automain
def run(_log, max_epochs, channels_to_use, dataloader_params, lr,
        data_folder, sleep_stages, arousal_annotation,
        weights_arousal, weights_sleep, model_name, data_name):
    # for saving log to file
    # fh = logging.FileHandler('spam.log')
    # _log.addHandler(fh)
    _log.setLevel(logging.INFO)

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # device = 'cpu'

    # Generators

    training_set, validation_set = get_dataloader(data_folder, model_name, data_name, size="default")
    training_generator = data.DataLoader(training_set, **dataloader_params)
    validation_generator = data.DataLoader(validation_set, **dataloader_params)

    dataloaders = {"train": training_generator, "val": validation_generator}

    # models
    model, inputs = initialize_model(model_name, channels_to_use=channels_to_use)
    model.to(device)

    # ignore classes unscored and undefined
    # Set weights for class arousal vs not arousal

    weights_arousal = torch.tensor(weights_arousal).float().to(device)
    criterion_arousal = nn.CrossEntropyLoss(weight=weights_arousal)

    weights_sleep = torch.tensor(weights_sleep).float().to(device)
    criterion_sleep = nn.CrossEntropyLoss(weight=weights_sleep)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print(torch.cuda.get_device_name(device=device))
    # Loop over epochs
    for epoch in range(max_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            true_array = np.empty(0)
            pred_array = np.empty(0)
            true_array_sleep = np.empty(0)
            pred_array_sleep = np.empty(0)

            running_loss = 0.0
            counters = 0
            for ID, inputs, annotations_arousal, annotations_sleep in dataloaders[phase]:

                # print(ID)
                # Transfer to GPU
                inputs = inputs.to(device)
                annotations_arousal = annotations_arousal.to(device)
                annotations_sleep = annotations_sleep.to(device)
                optimizer.zero_grad()
                # print(inputs.shape, annotations_arousal.shape, annotations_sleep.shape)
                # originally only use first 12, ignore cardiogram?
                # ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'Chin1-Chin2', 'ABD', 'CHEST', 'AIRFLOW', 'SaO2', 'ECG']
                inputs = inputs[:, 0:channels_to_use, :]

                batch_sz_ = inputs.shape[0]
                data_sz = inputs.shape[-1]

                with torch.set_grad_enabled(phase == 'train'):

                    inputs = inputs.view([batch_sz_, channels_to_use, data_sz]).type(torch.cuda.FloatTensor).to(device)
                    annotations_arousal = annotations_arousal.type(torch.cuda.LongTensor).to(device)
                    annotations_sleep = annotations_sleep.type(torch.cuda.LongTensor).to(device)

                    arousal_out, sleep_out = model.forward(inputs)
                    # print(arousal_out.shape, sleep_out.shape)
                    loss_arousal = criterion_arousal(arousal_out, annotations_arousal)
                    loss_sleep = criterion_sleep(sleep_out, annotations_sleep)
                    loss = loss_arousal + loss_sleep
                    # print(epoch, phase, "Loss ", loss.item())
                    # print("Max Mem GB  ", torch.cuda.max_memory_allocated(device=device) * 1e-9)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                true_array_ = annotations_arousal.cpu().numpy().squeeze().astype(int)
                pred_array_ = arousal_out.argmax(dim=1).cpu().numpy().squeeze().astype(int)


                # remove all 0 (unscored) ORDER of pred/True Matters
                pred_array = np.append(pred_array, pred_array_[true_array_ != 0])
                true_array = np.append(true_array, true_array_[true_array_ != 0])

                pred_array_sleep_ = sleep_out.argmax(dim=1).cpu().numpy().squeeze().astype(int)
                true_array_sleep_ = annotations_sleep.cpu().numpy().squeeze().astype(int)
                # remove all 4 (undefined) ORDER of pred/True Matters
                pred_array_sleep = np.append(pred_array_sleep, pred_array_sleep_[true_array_sleep_ != 4])
                true_array_sleep = np.append(true_array_sleep, true_array_sleep_[true_array_sleep_ != 4])

                running_loss += loss.item() * batch_sz_

                # print("acc", accuracy(pred_array, true_array))
                # print("Max Mem GB  ", torch.cuda.max_memory_allocated(device=device) * 1e-9)
                # print()
                counters += 1
                if counters == 50:
                    break

            print("Max Mem GB  ", torch.cuda.max_memory_allocated(device=device) * 1e-9)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            acc = accuracy(pred_array, true_array)
            acc_sleep = accuracy(pred_array_sleep, true_array_sleep)

            writer.add_scalar('{}_Loss'.format(phase), epoch_loss, global_step=epoch)
            ex.log_scalar('{}_Loss'.format(phase), epoch_loss, epoch)
            # writer.add_scalar('{}_Accuracy_arousal'.format(phase), acc, global_step=epoch)
            # ex.log_scalar('{}_Accuracy_arousal'.format(phase), acc, epoch)
            # writer.add_scalar('{}_Accuracy_sleep_staging'.format(phase), acc_sleep, global_step=epoch)
            # ex.log_scalar('{}_Accuracy_sleep_staging'.format(phase), acc_sleep, epoch)

            # todo
            balanced_arousal = metrics.balanced_accuracy_score(true_array, pred_array)
            balanced_sleep = metrics.balanced_accuracy_score(true_array, pred_array)

            writer.add_scalar('{}_Accuracy_arousal'.format(phase), balanced_arousal, global_step=epoch)
            ex.log_scalar('{}_Accuracy_arousal'.format(phase), balanced_arousal, epoch)
            writer.add_scalar('{}_Accuracy_sleep_staging'.format(phase), balanced_sleep, global_step=epoch)
            ex.log_scalar('{}_Accuracy_sleep_staging'.format(phase), balanced_sleep, epoch)

            report = metrics.classification_report(true_array,
                                                   pred_array,
                                                   labels=[0, 1, 2],
                                                   target_names=arousal_annotation,
                                                   zero_division=0)
            writer.add_text('Report Arousals {}'.format(phase), report + '\n', global_step=epoch)
            sleep_report = metrics.classification_report(true_array_sleep,
                                                         pred_array_sleep,
                                                         labels=[0, 1, 2, 3, 4, 5],
                                                         target_names=sleep_stages,
                                                         zero_division=0)
            writer.add_text('Report Sleep staging {}'.format(phase), sleep_report + '\n', global_step=epoch)

            if True:
                _log.info("\n{}: epoch: {} Loss {:.3f} Accuracy arousal: {:.3f} Accuracy Sleep: {:.3f}\n".format(phase, epoch, epoch_loss, acc, acc_sleep))
                _log.info("\n{}: Arousal Report: \n{}\n".format(phase, report))
                _log.info("\n{}: Sleep Report: \n{}\n".format(phase, sleep_report))


    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # print(y_test, "\n", y_pred)

    # Plot normalized confusion matrix
    # fig_confusion = plot_confusion_matrix(true_array, pred_array, classes=arousal_annotation, normalize=True, title='Normalized confusion matrix')
    # writer.add_figure("Confusion Matrix", fig_confusion)
    #
    # fig_classes = plot_classes_distribution(labels, classes=arousal_annotation)
    # writer.add_figure("Class distribution", fig_classes)

    # writer.close()
