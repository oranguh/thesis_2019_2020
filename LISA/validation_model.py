import os
import pickle as pkl
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.utils import data
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from tools import accuracy, CustomFormatter, get_dataloader
from score2018 import Challenge2018Score
from visualizations import plot_confusion_matrix, plot_classes_distribution


def main():

    channel_index = 0
    pre_traineds = []
    # pre_traineds += sorted(Path("models/weights").glob("*SHHSweights*"))
    # pre_traineds += sorted(Path("models/weights").glob("*snoozeweights*"))
    # pre_traineds += sorted(Path("models/channels").glob("*"))
    # pre_traineds += sorted(Path("models/weights").glob("*"))
    # pre_traineds += sorted(Path("models/sleep_staging").glob("*"))

    # pre_traineds += sorted(Path("models/weights").glob("*1_100"))
    # pre_traineds += sorted(Path("models/weights").glob("*1_200"))
    # pre_traineds += sorted(Path("models/combined_dataset").glob("*sleep*"))
    pre_traineds += sorted(Path("models/frequency").glob("*"))
    # print(pre_traineds)
    for model_name in ["Deep_Sleep"]:
        # model_name = "ConvNet_IID"
        # for howe use batchsize 6 on lisa?
        # model_name = "Howe_Patterson"
        # model_name = "Deep_Sleep"
        pass

    for pre_trained_model in pre_traineds:
        for data_name in ["snooze", "SHHS"]:
        # for data_name in ["snooze"]:

            comment = pre_trained_model.name + "_to_" + data_name
            validate(data_name, model_name, pre_trained_model, channel_index, comment)

        if False:
            if data_name == "SHHS":
                for channel_index, channel in enumerate(["C3_A2", "C4_A1"]):
                    comment = pre_trained_model.name + "_to_" + data_name + "_" + channel
                    validate(data_name, model_name, pre_trained_model, channel_index, comment)

            elif data_name == "snooze":
                for channel_index, channel in enumerate(["F3_M2", "F4_M1", "C3_M2", "C4_M1"]):
                    comment = pre_trained_model.name + "_to_" + data_name + "_" + channel
                    validate(data_name, model_name, pre_trained_model, channel_index, comment)
            else:
                print("data not found")

            if False:
                comment = pre_trained_model.name + "_to_" + data_name
                # comment = pre_trained_model.name

                channel_index = 0
                model_name = "Deep_Sleep"
                print(pre_trained_model)
                validate(data_name, model_name, pre_trained_model, channel_index, comment)

    # SHHS
    #     EEG (sec): 	C3 	A2
    #     EEG:    C4 	A1

    # Snooze
    # F3-M2, F4-M1, C3-M2, C4-M1, O1-M2 and O2-M1; one electrooculography (EOG) signal at E1-M2;
    # three electromyography (EMG) signals of chin, abdominal and chest movements; one measure of respiratory
    # airflow; one measure of oxygen saturation (SaO2); one electrocardiogram (ECG)

    # Philips
    # (Fpz - M2; Fpz - Fp1)


def validate(data_name, model_name, pre_trained_model, channel_index, comment):

    Challenge2018Scorer = Challenge2018Score()

    model = torch.load(pre_trained_model)
    weights_sleep = None
    weights_arousal = None

    if platform.system() == 'Windows':
        if data_name == "SHHS":
            data_folder = 'K:\\shhs\\polysomnography\\shh1_numpy'
            weights_sleep = [.3, .05, .3, .3, .0, .05]  # for SHHS
            weights_arousal = [.0, .05, .95]  # SHHS
        else:
            data_folder = 'F:\\you-snooze-you-win-the-physionet-computing-in-cardiology-challenge-2018-1.0.0\\marco'
            data_folder = 'D:\\data\\snooze\\marco'
            weights_sleep = [.2, .1, .3, .2, .0, .2]  # for Snooze
            weights_arousal = [.0, .05, .95]  # Snooze

        # Parameters for dataloader
        dataloader_params = {'batch_size': 4,
                             'shuffle': False,
                             'num_workers': 0}
    else:
        if data_name == "SHHS":
            data_folder = '/project/marcoh/shhs/polysomnography/shh1_numpy/'
        else:
            data_folder = '/project/marcoh/you_snooze_you_win/marco/'

        # Parameters for dataloader
        dataloader_params = {'batch_size': 3,
                             'shuffle': True,
                             'num_workers': 4}

    writer_path = os.path.join("validations", comment)
    writer = SummaryWriter(log_dir=writer_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, validation_set = get_dataloader(data_folder, model_name, data_name, size="default")
    validation_generator = data.DataLoader(validation_set, **dataloader_params)

    dataloaders = {"val": validation_generator}
    phase = "val"

    weights_arousal = torch.tensor(weights_arousal).float().to(device)
    criterion_arousal = nn.CrossEntropyLoss(weight=weights_arousal)

    weights_sleep = torch.tensor(weights_sleep).float().to(device)
    criterion_sleep = nn.CrossEntropyLoss(weight=weights_sleep)

    channels_to_use = 1

    model.eval()

    accuracies = np.empty(0)
    kappas = np.empty(0)

    true_array = np.empty(0)
    pred_array = np.empty(0)
    true_array_sleep = np.empty(0)
    pred_array_sleep = np.empty(0)

    running_loss = 0.0
    counters = 0
    epoch = 0

    for ID, inputs, annotations_arousal, annotations_sleep in dataloaders[phase]:
        with torch.set_grad_enabled(False):

            inputs = inputs.to(device)

            if channel_index == "random":
                if data_name == "SHHS":
                    max_chan = 2
                elif data_name == "snooze":
                    max_chan = 4
                else:
                    pass
                rand_channel = np.random.randint(0, max_chan)
                inputs = inputs[:, rand_channel, :]
            else:
                inputs = inputs[:, channel_index, :]


            m = torch.mean(inputs.float())
            s = torch.std(inputs.float())
            inputs = inputs - m
            inputs = inputs / s

            annotations_arousal = annotations_arousal.to(device)
            annotations_sleep = annotations_sleep.to(device)

            batch_sz_ = inputs.shape[0]
            data_sz = inputs.shape[-1]

            inputs = inputs.view([batch_sz_, channels_to_use, data_sz]).type(torch.cuda.FloatTensor).to(device)
            annotations_arousal = annotations_arousal.type(torch.cuda.LongTensor).to(device)
            annotations_sleep = annotations_sleep.type(torch.cuda.LongTensor).to(device)

            arousal_out, sleep_out = model.forward(inputs)
            # print(arousal_out.shape, sleep_out.shape)
            # print(annotations_arousal.shape, annotations_sleep.shape)
            loss_arousal = criterion_arousal(arousal_out, annotations_arousal)
            loss_sleep = criterion_sleep(sleep_out, annotations_sleep)
            loss = 5 * loss_arousal + loss_sleep
            # print(epoch, phase, "Loss ", loss.item())
            # print("Max Mem GB  ", torch.cuda.max_memory_allocated(device=device) * 1e-9)

            true_array_ = annotations_arousal.cpu().numpy().squeeze().astype(int)
            pred_array_ = arousal_out.argmax(dim=1).cpu().numpy().squeeze().astype(int)

            prediction_prob_arousal = torch.nn.functional.softmax(arousal_out, dim=1)[:, 2,
                                      :].detach().cpu().numpy().squeeze()

            do_auroc(Challenge2018Scorer, true_array_, prediction_prob_arousal, ID, comment, epoch)

            # set all 0 (unscored) to predictions to 1 (not-arousal)
            pred_array_[pred_array_ == 0] = 1

            accuracies_, kappas_ = do_stats_arousal(true_array_, pred_array_, ID)
            accuracies = np.append(accuracies, accuracies_)
            kappas = np.append(kappas, kappas_)

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

            counters += 1

            if counters == 4:
                print("Max Mem GB  ", torch.cuda.max_memory_allocated(device=device) * 1e-9)
                # save_metrics(running_loss, dataloaders, phase, pred_array, true_array, pred_array_sleep, true_array_sleep,
                #              Challenge2018Scorer, writer, epoch, comment)
                epoch += 1

                true_array = np.empty(0)
                pred_array = np.empty(0)
                true_array_sleep = np.empty(0)
                pred_array_sleep = np.empty(0)

                print(Challenge2018Scorer.gross_auprc(), Challenge2018Scorer.gross_auroc())

                del Challenge2018Scorer
                Challenge2018Scorer = Challenge2018Score()

                running_loss = 0.0
                counters = 0

            del arousal_out
            del sleep_out
            del loss_arousal
            del loss_sleep
            plt.close("all")

        if epoch == 10:
            Path("statistics").mkdir(parents=True, exist_ok=True)
            Path("statistics/accuracies").mkdir(parents=True, exist_ok=True)
            Path("statistics/kappas").mkdir(parents=True, exist_ok=True)
            np.save(Path("statistics/accuracies") / comment, accuracies)
            np.save(Path("statistics/kappas") / comment, kappas)
            break
    print("END")

def save_metrics(running_loss, dataloaders, phase, pred_array, true_array, pred_array_sleep, true_array_sleep,
                 Challenge2018Scorer, writer, epoch, comment):

    epoch_loss = running_loss / len(dataloaders[phase].dataset)
    acc = accuracy(pred_array, true_array)
    acc_sleep = accuracy(pred_array_sleep, true_array_sleep)

    writer.add_scalar('Loss/{}'.format(phase), epoch_loss, global_step=epoch)
    # ex.log_scalar('Loss/{}'.format(phase), epoch_loss, epoch)
    writer.add_scalar('Accuracy_arousal/{}'.format(phase), acc, global_step=epoch)
    # ex.log_scalar('Accuracy_arousal/{}'.format(phase), acc, epoch)
    writer.add_scalar('Accuracy_sleep_staging/{}'.format(phase), acc_sleep, global_step=epoch)
    # ex.log_scalar('Accuracy_sleep_staging/{}'.format(phase), acc_sleep, epoch)

    AUROC_arousal = Challenge2018Scorer.gross_auroc()
    writer.add_scalar('AUROC_arousal/{}'.format(phase), AUROC_arousal, global_step=epoch)
    # ex.log_scalar('AUROC_arousal/{}'.format(phase), AUROC_arousal, epoch)

    AUPRC_arousal = Challenge2018Scorer.gross_auprc()
    writer.add_scalar('AUPRC_arousal_arousal/{}'.format(phase), AUPRC_arousal, global_step=epoch)
    # ex.log_scalar('AUPRC_arousal_arousal/{}'.format(phase), AUPRC_arousal, epoch)

    # print("PRC", Challenge2018Scorer.gross_auprc(), metrics.average_precision_score(true_array, pred_array))
    # print("ROC", Challenge2018Scorer.gross_auroc(), metrics.roc_auc_score(true_array, pred_array))

    balanced_arousal = metrics.balanced_accuracy_score(true_array, pred_array)
    balanced_sleep = metrics.balanced_accuracy_score(true_array_sleep, pred_array_sleep)
    writer.add_scalar('Balanced_Accuracy_arousal/{}'.format(phase), balanced_arousal, global_step=epoch)
    # ex.log_scalar('Balanced_Accuracy_arousal/{}'.format(phase), balanced_arousal, epoch)
    writer.add_scalar('Balanced_Accuracy_sleep_staging/{}'.format(phase), balanced_sleep, global_step=epoch)
    # ex.log_scalar('Balanced_Accuracy_sleep_staging/{}'.format(phase), balanced_sleep, epoch)

    cohen_kappa_arousal = metrics.cohen_kappa_score(true_array, pred_array, labels=[0, 1, 2])
    writer.add_scalar('Kappa_arousal/{}'.format(phase), cohen_kappa_arousal, global_step=epoch)
    cohen_kappa_sleep = metrics.cohen_kappa_score(true_array_sleep, pred_array_sleep, labels=[0, 1, 2, 3, 4, 5])
    writer.add_scalar('Kappa_sleep/{}'.format(phase), cohen_kappa_sleep, global_step=epoch)

    arousal_annotation = ["not_scored", "not_arousal", "Arousal"]
    sleep_stages = ['\tundefined', '\tnonrem1', '\tnonrem2', '\tnonrem3', '\trem', '\twake']

    report = metrics.classification_report(true_array,
                                           pred_array,
                                           labels=[0, 1, 2],
                                           target_names=arousal_annotation,
                                           output_dict=True,
                                           zero_division=0)

    df = pd.DataFrame.from_dict(report, orient='index', dtype="object")
    df = df.astype(dtype={"precision": "float64", "recall": "float64", "f1-score": "float64", "support": "object"})
    df = df.round(2)
    writer.add_text('Report Arousals/{}'.format(phase), df.to_markdown(), global_step=epoch)

    sleep_report = metrics.classification_report(true_array_sleep,
                                                 pred_array_sleep,
                                                 labels=[4, 0, 1, 2, 3, 5],
                                                 target_names=sleep_stages,
                                                 output_dict=True,
                                                 zero_division=0)
    del sleep_report["accuracy"]
    df2 = pd.DataFrame.from_dict(sleep_report, orient='index', dtype="object")
    df2 = df2.astype(dtype={"precision": "float64", "recall": "float64", "f1-score": "float64", "support": "object"})
    df2 = df2.round(2)

    writer.add_text('Report Sleep staging/{}'.format(phase), df2.to_markdown(), global_step=epoch)

    print("\n{}: epoch: {} Loss {:.3f} Accuracy arousal: {:.3f} Accuracy Sleep: {:.3f}\n".format(phase, epoch, epoch_loss, acc, acc_sleep))
    print("\n{}: Arousal Report: \n{}\n".format(phase, df.to_markdown()))

    print("\n{}: Sleep Report: \n{}\n".format(phase, df2.to_markdown()))

    # Plot normalized confusion matrix
    fig_confusion = plot_confusion_matrix(true_array, pred_array, classes=["not arousal", "arousal"], title=comment+"_"+str(epoch), labels=[1, 2])
    writer.add_figure("Confusion Matrix/{}".format(phase), fig_confusion, global_step=epoch)

    plt.close("all")

def load_obj(name):
    with open(name, 'rb') as f:
        return pkl.load(f)

def do_stats_arousal(true, pred, id):

    accuracies = []
    kappas = []

    for i, _ in enumerate(true):
        pred_record = pred[i][true[i] != 0]
        true_record = true[i][true[i] != 0]
        accuracies.append(metrics.balanced_accuracy_score(true_record, pred_record))
        kappas.append(metrics.cohen_kappa_score(true_record, pred_record, labels=[0, 1, 2]))

    return accuracies, kappas

def do_auroc(Challenge2018Scorer, true_array_, prediction_prob_arousal, ID, comment, epoch):

    fpr, tpr = 0, 0
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('{}'.format(comment), fontsize=16)
    lw = 2
    ax[0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    for i, _ in enumerate(true_array_):
        true = true_array_[i]
        pred = prediction_prob_arousal[i]

        Challenge2018Scorer.score_record(true[true != 0] - 1, pred[true != 0], ID[i])

        # print(Challenge2018Scorer._record_auc)

        fpr, tpr, _ = metrics.roc_curve(true[true != 0] - 1, pred[true != 0])
        prec, recall, _ = metrics.precision_recall_curve(true[true != 0] - 1, pred[true != 0], pos_label=1)

        # print(fpr.shape, recall.shape, true[true != 0].shape)

        roc_auc = metrics.auc(fpr, tpr)
        prc_auc = metrics.auc(recall, prec)



        ax[0].plot(fpr, tpr, color='darkorange',
                 lw=lw, label='{} (area = {:0.2f})'.format(ID[i], roc_auc), alpha=0.3)
        ax[0].set_xlim([0.0, 1.0])
        ax[0].set_ylim([0.0, 1.05])
        ax[0].set_xlabel('False Positive Rate')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].set_title('Receiver operating characteristic')
        ax[0].legend(loc="lower right")

        ax[1].plot(recall, prec, color='darkorange',
                 lw=lw, label='{} (area = {:0.2f})'.format(ID[i], prc_auc), alpha=0.3)
        ax[1].set_xlim([0.0, 1.0])
        ax[1].set_ylim([0.0, 1.05])
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precision')
        ax[1].set_title('Precision Recall curve')
        ax[1].legend(loc="upper right")

        # plt.hold(True)
    Path("figures/aurocs").mkdir(parents=True, exist_ok=True)
    filename = comment + str(epoch) + '.png'
    plt.savefig(Path("figures/aurocs") / filename)
    plt.clf()

if __name__ == '__main__':
    main()
