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
from score2018 import Challenge2018Score
import platform
import time

def initialize_model(model_name, channels_to_use):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    #

    if model_name == "ConvNet_IID":
        """ ConvNet_IID
        """
        model_ft = model = ConvNet_IID(channels_to_use=channels_to_use)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "Howe_Patterson":
        """ Howe_Patterson
        """
        model_ft = Howe_Patterson(channels_to_use=channels_to_use)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)


    elif model_name == "Deep_Sleep":
        """ Deep_Sleep
        """
        # largest recording length 3422800 at 100Hz

        model_ft = Deep_Sleep(channels_to_use=channels_to_use)
        # set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

def load_obj(name):
    with open(name, 'rb') as f:
        return pkl.load(f)


ex = Experiment("hello")
ex.observers.append(FileStorageObserver('my_runs'))
ex.observers.__dir__()
Challenge2018Scorer = Challenge2018Score()

@ex.config
def cfg():

    weights_sleep, weights_arousal = [.2, .2, .2, .2, .0, .2], [.0, .5, .5] # default weights?

    # models to train... Howe_Patterson, U-Net, CNN, cNN + LSTM, wavenet? Model also determines the way the data is loaded.

    # for conv use 2048 batchsize ["ConvNet_IID", "Howe_Patterson"]
    # model_name = "ConvNet_IID"
    # for howe use batchsize 6 on lisa?
    # model_name = "Howe_Patterson"
    model_name = "Deep_Sleep"

    # Dataset to use.... also determines path. Snooze, SHHS, Philips, HMC
    # data_name = "snooze"
    data_name = "SHHS"
    # Predicting future labels is a thing I want to focus on.
    # pred_future = True
    pretrained = False
    # pretrained = "models\\Jul09_23-59-04_BananaDeep_Sleep_SHHS"
    # pretrained = "models\\Jul08_00-46-48_DESKTOP-5TLTVUT20200708-004648"
    # pretrained = "models\\Jul15_18-22-19_BananaDeep_Sleep_SHHS"

    comment = "C3_A2_(sec)_norm_off"

    if platform.system() == 'Windows':
        if data_name == "SHHS":
            data_folder = 'K:\\shhs\\polysomnography\\shh1_numpy'
            weights_sleep = [.3, .05, .3, .3, .0, .05]  # for SHHS
            weights_arousal = [.0, .05, .95]  # SHHS
        elif data_name == "snooze":
            data_folder = 'F:\\you-snooze-you-win-the-physionet-computing-in-cardiology-challenge-2018-1.0.0\\marco'
            data_folder = 'D:\\data\\snooze\\marco'
            weights_sleep = [.2, .1, .3, .2, .0, .2]  # for Snooze
            weights_arousal = [.0, .05, .95]  # Snooze
        else:
            print("data not found")

        # Parameters for dataloader
        dataloader_params = {'batch_size': 3,
                             'shuffle': True,
                             'num_workers': 1}
    else:
        if data_name == "SHHS":
            data_folder = '/project/marcoh/shhs/polysomnography/shh1_numpy/'
        elif data_name == "snooze":
            data_folder = '/project/marcoh/you_snooze_you_win/marco/'
        else:
            print("data not found")
        # Parameters for dataloader
        dataloader_params = {'batch_size': 3,
                             'shuffle': True,
                             'num_workers': 4}

    max_epochs = 30
    # full PSG has 12, we use 1, always
    channels_to_use = 1

    optimizer_params = {"lr": 1e-4,
                        "optim": "Adam"}
    lr = 1e-4
    channel_id = 0
    sleep_stage_importance = 0

def training(_log, max_epochs, channels_to_use, dataloader_params, lr,
        data_folder, weights_arousal, weights_sleep, model_name, data_name, pretrained, comment,
             channel_id, sleep_stage_importance):
    # writer = SummaryWriter(comment=str(datetime.now().strftime("%Y%m%d-%H%M%S")))
    writer = SummaryWriter(comment=model_name + "_" + data_name + "_" + comment)

    print(comment)
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
    if pretrained:
        model = torch.load(pretrained)
        print("Pre-trained model loaded \n", model)
    else:
        model = initialize_model(model_name, channels_to_use=channels_to_use)
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
            Challenge2018Scorer._pos_values, Challenge2018Scorer._neg_values = 0, 0

            running_loss = 0.0
            counters = 0
            for ID, inputs, annotations_arousal, annotations_sleep in dataloaders[phase]:
                # print(ID, inputs.shape, annotations_arousal.shape, annotations_sleep.shape)

                # Transfer to GPU
                inputs = inputs.to(device)
                annotations_arousal = annotations_arousal.to(device)
                annotations_sleep = annotations_sleep.to(device)
                optimizer.zero_grad()
                # print(inputs.shape, annotations_arousal.shape, annotations_sleep.shape)
                # originally only use first 12, ignore cardiogram?

                # SHHS [C3-A2 (sec), C4-A1]
                # snooze ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'Chin1-Chin2', 'ABD', 'CHEST', 'AIRFLOW', 'SaO2', 'ECG']

                # pick a channel, or do random channel

                if channel_id == "random":
                    if data_name == "SHHS":
                        max_chan = 2
                    elif data_name == "snooze":
                        max_chan = 4
                    else:
                        pass
                    rand_channel = np.random.randint(0, max_chan)
                    inputs = inputs[:, rand_channel, :]
                else:
                    inputs = inputs[:, channel_id, :]

                m = torch.mean(inputs.float())
                s = torch.std(inputs.float())
                inputs = inputs - m
                inputs = inputs / s

                batch_sz_ = inputs.shape[0]
                data_sz = inputs.shape[-1]

                with torch.set_grad_enabled(phase == 'train'):

                    inputs = inputs.view([batch_sz_, channels_to_use, data_sz]).type(torch.cuda.FloatTensor).to(device)
                    annotations_arousal = annotations_arousal.type(torch.cuda.LongTensor).to(device)
                    annotations_sleep = annotations_sleep.type(torch.cuda.LongTensor).to(device)

                    arousal_out, sleep_out = model.forward(inputs)
                    # print(arousal_out.shape, sleep_out.shape)
                    # print(annotations_arousal.shape, annotations_sleep.shape)
                    loss_arousal = criterion_arousal(arousal_out, annotations_arousal)
                    loss_sleep = criterion_sleep(sleep_out, annotations_sleep)

                    loss = loss_arousal + sleep_stage_importance * loss_sleep
                    # loss = loss_arousal
                    # print(epoch, phase, "Loss ", loss.item())
                    # print("Max Mem GB  ", torch.cuda.max_memory_allocated(device=device) * 1e-9)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                true_array_ = annotations_arousal.cpu().numpy().squeeze().astype(int)
                pred_array_ = arousal_out.argmax(dim=1).cpu().numpy().squeeze().astype(int)

                if arousal_out.dim() == 3:
                    prediction_prob_arousal = torch.nn.functional.softmax(arousal_out, dim=1)[:, 2, :]\
                        .detach().cpu().numpy().squeeze()
                else:
                    prediction_prob_arousal = torch.nn.functional.softmax(arousal_out, dim=1)[:, 2]\
                        .detach().cpu().numpy().squeeze()
                Challenge2018Scorer.score_record(true_array_[true_array_ != 0] - 1,
                                                 prediction_prob_arousal[true_array_ != 0])

                # set all 0 (unscored) to predictions to 1 (not-arousal)
                pred_array_[pred_array_ == 0] = 1
                # remove all 0 (unscored) ORDER of pred/True Matters
                pred_array = np.append(pred_array, pred_array_[true_array_ != 0])
                true_array = np.append(true_array, true_array_[true_array_ != 0])

                pred_array_sleep_ = sleep_out.argmax(dim=1).cpu().numpy().squeeze().astype(int)
                true_array_sleep_ = annotations_sleep.cpu().numpy().squeeze().astype(int)
                # remove all 4 (undefined) ORDER of pred/True Matters
                # TODO this might become very very large if doing 2**22 length records per sample
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
            model_savepath = os.path.split(writer.log_dir)[-1]
            model_savepath = os.path.join("models", model_savepath)
            torch.save(model, model_savepath)

            def save_metrics(arousal_pred, arousal_true, sleep_pred, sleep_true, epoch_loss, writer, epoch):
                pass
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            acc = accuracy(pred_array, true_array)
            acc_sleep = accuracy(pred_array_sleep, true_array_sleep)

            writer.add_scalar('Loss/{}'.format(phase), epoch_loss, global_step=epoch)
            ex.log_scalar('Loss/{}'.format(phase), epoch_loss, epoch)

            writer.add_scalar('Accuracy_arousal/{}'.format(phase), acc, global_step=epoch)
            ex.log_scalar('Accuracy_arousal/{}'.format(phase), acc, epoch)
            writer.add_scalar('Accuracy_sleep_staging/{}'.format(phase), acc_sleep, global_step=epoch)
            ex.log_scalar('Accuracy_sleep_staging/{}'.format(phase), acc_sleep, epoch)

            AUROC_arousal = Challenge2018Scorer.gross_auroc()
            writer.add_scalar('AUROC_arousal/{}'.format(phase), AUROC_arousal, global_step=epoch)
            ex.log_scalar('AUROC_arousal/{}'.format(phase), AUROC_arousal, epoch)

            AUPRC_arousal = Challenge2018Scorer.gross_auprc()
            writer.add_scalar('AUPRC_arousal_arousal/{}'.format(phase), AUPRC_arousal, global_step=epoch)
            ex.log_scalar('AUPRC_arousal_arousal/{}'.format(phase), AUPRC_arousal, epoch)



            print("\n\nEpoch ", epoch, "phase: ", phase)
            print("PRC ", AUPRC_arousal)
            print("ROC ", AUROC_arousal)
            print("loss ", loss.item())
            print("acc arousal ", acc)
            print("acc sleep ", acc_sleep)
            if epoch % 10 == 0:
                report = metrics.classification_report(true_array,
                                                       pred_array,
                                                       labels=[0, 1, 2],
                                                       target_names=["not_scored", "not_arousal", "Arousal"],
                                                       zero_division=0)
                writer.add_text('Report Arousals/{}'.format(phase), report + '\n', global_step=epoch)
                print(report)

            if False:
                balanced_arousal = metrics.balanced_accuracy_score(true_array, pred_array)
                balanced_sleep = metrics.balanced_accuracy_score(true_array_sleep, pred_array_sleep)
                writer.add_scalar('Balanced_Accuracy_arousal/{}'.format(phase), balanced_arousal, global_step=epoch)
                ex.log_scalar('Balanced_Accuracy_arousal/{}'.format(phase), balanced_arousal, epoch)
                writer.add_scalar('Balanced_Accuracy_sleep_staging/{}'.format(phase), balanced_sleep, global_step=epoch)
                ex.log_scalar('Balanced_Accuracy_sleep_staging/{}'.format(phase), balanced_sleep, epoch)

                cohen_kappa_arousal = metrics.cohen_kappa_score(true_array, pred_array, labels=[0, 1, 2])
                cohen_kappa_sleep = metrics.cohen_kappa_score(true_array_sleep, pred_array_sleep, labels=[0, 1, 2, 3, 4, 5])
                writer.add_scalar('Kappa_arousal/{}'.format(phase), cohen_kappa_arousal, global_step=epoch)
                writer.add_scalar('Kappa_sleep/{}'.format(phase), cohen_kappa_sleep, global_step=epoch)

                report = metrics.classification_report(true_array,
                                                       pred_array,
                                                       labels=[0, 1, 2],
                                                       target_names=["not_scored", "not_arousal", "Arousal"],
                                                       zero_division=0)
                writer.add_text('Report Arousals/{}'.format(phase), report + '\n', global_step=epoch)
                sleep_report = metrics.classification_report(true_array_sleep,
                                                             pred_array_sleep,
                                                             labels=[0, 1, 2, 3, 4, 5],
                                                             target_names=['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake'],
                                                             zero_division=0)
                writer.add_text('Report Sleep staging/{}'.format(phase), sleep_report + '\n', global_step=epoch)


                _log.info("\n{}: epoch: {} Loss {:.3f} Accuracy arousal: {:.3f} Accuracy Sleep: {:.3f}\n".format(phase, epoch, epoch_loss, acc, acc_sleep))
                _log.info("\n{}: Arousal Report: \n{}\n".format(phase, report))
                _log.info("\n{}: Sleep Report: \n{}\n".format(phase, sleep_report))


    # np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # print(y_test, "\n", y_pred)

    # Plot normalized confusion matrix
    # fig_confusion = plot_confusion_matrix(true_array, pred_array, classes=arousal_annotation, normalize=True, title='Normalized confusion matrix')
    # writer.add_figure("Confusion Matrix", fig_confusion)
    #
    # fig_classes = plot_classes_distribution(labels, classes=arousal_annotation)
    # writer.add_figure("Class distribution", fig_classes)

    # writer.close()

@ex.automain
def run(_log, max_epochs, channels_to_use, dataloader_params, lr,
        data_folder, weights_arousal, weights_sleep, model_name, data_name, pretrained, comment, channel_id, sleep_stage_importance):


    data_name = "SHHS"
    data_folder = 'K:\\shhs\\polysomnography\\shh1_numpy'
    # """" Experiments with weights"""
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .005, .995], weights_sleep, model_name, data_name, pretrained, "weights_1_200",
    #     channel_id, sleep_stage_importance)
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .01, .99], weights_sleep, model_name, data_name, pretrained, "weights_1_100",
    #     channel_id, sleep_stage_importance)
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .02, .98], weights_sleep, model_name, data_name, pretrained, "weights_1_50",
    #     channel_id, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .05, .95], weights_sleep, model_name, data_name, pretrained, "weights_1_20",
    #     channel_id, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "weights_1_10",
    #     channel_id, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .2, .8], weights_sleep, model_name, data_name, pretrained, "weights_1_5",
    #     channel_id, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .5, .5], weights_sleep, model_name, data_name, pretrained, "weights_1_1",
    #     channel_id, sleep_stage_importance)
    #
    data_name = "snooze"
    data_folder = 'D:\\data\\snooze\\marco'
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .005, .995], weights_sleep, model_name, data_name, pretrained, "weights_1_200",
    #     channel_id, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .01, .99], weights_sleep, model_name, data_name, pretrained, "weights_1_100",
    #     channel_id, sleep_stage_importance)

    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .02, .98], weights_sleep, model_name, data_name, pretrained, "weights_1_50",
    #     channel_id, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .05, .95], weights_sleep, model_name, data_name, pretrained, "weights_1_20",
    #     channel_id, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "weights_1_10",
    #     channel_id, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .2, .8], weights_sleep, model_name, data_name, pretrained, "weights_1_5",
    #     channel_id, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #     data_folder, [.0, .5, .5], weights_sleep, model_name, data_name, pretrained, "weights_1_1",
    #     channel_id, sleep_stage_importance)
    #
    """ channel experiments ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1'] """
    data_name = "snooze"
    data_folder = 'D:\\data\\snooze\\marco'
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "random_channel",
    #          "random", sleep_stage_importance)

    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "F3-M2",
    #          0, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "F4-M1",
    #          1, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "C3-M2",
    #          2, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "C4-M1",
    #          3, sleep_stage_importance)

    data_name = "SHHS"
    data_folder = 'K:\\shhs\\polysomnography\\shh1_numpy'
    """" # SHHS [C3-A2 (sec), C4-A1] """
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "random_channel",
    #          "random", sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "C3-A2",
    #          0, sleep_stage_importance)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "C4-A1",
    #          1, sleep_stage_importance)

    data_name = "SHHS"
    data_folder = 'K:\\shhs\\polysomnography\\shh1_numpy'
    """" Adding sleep staging """
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "sleep_1_1",
    #          0, 1)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "sleep_1_2",
    #          0, 0.5)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "sleep_1_5",
    #          0, 0.2)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "sleep_1_10",
    #          0, 0.1)

    # data_name = "snooze"
    # data_folder = 'D:\\data\\snooze\\marco'
    """" Adding sleep staging """
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "sleep_1_1",
    #          0, 1)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "sleep_1_2",
    #          0, 0.5)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "sleep_1_5",
    #          0, 0.2)
    #
    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, "sleep_1_10",
    #          0, 0.1)

    # model_name = "ConvNet_IID"
    # for howe use batchsize 6 on lisa?
    # model_name = "Howe_Patterson"
    model_name = "Deep_Sleep"

    # data_name = "combined"
    comment = "50Hz"
    # data_folder = ['D:\\data\\snooze\\marco', 'K:\\shhs\\polysomnography\\shh1_numpy']

    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, comment,
    #          0, sleep_stage_importance)



    """
    CONVNET EXPIRIMENTS
    """
    dataloader_params = {'batch_size': 300,
                         'shuffle': True,
                         'num_workers': 2}

    model_name = "ConvNet_IID"
    comment = ""
    data_name = "SHHS"
    data_folder = 'K:\\shhs\\polysomnography\\shh1_numpy'

    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, comment,
    #          0, sleep_stage_importance)

    dataloader_params = {'batch_size': 300,
                         'shuffle': True,
                         'num_workers': 2}

    data_name = "snooze"
    data_folder = 'D:\\data\\snooze\\marco'

    training(_log, max_epochs, channels_to_use, dataloader_params, lr,
             data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, comment,
             0, sleep_stage_importance)

    dataloader_params = {'batch_size': 50,
                         'shuffle': True,
                         'num_workers': 2}
    data_name = "combined"
    data_folder = ['D:\\data\\snooze\\marco', 'K:\\shhs\\polysomnography\\shh1_numpy']

    # training(_log, max_epochs, channels_to_use, dataloader_params, lr,
    #          data_folder, [.0, .1, .9], weights_sleep, model_name, data_name, pretrained, comment,
    #          0, sleep_stage_importance)