import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

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
import matplotlib
import seaborn as sns

from tools import accuracy, CustomFormatter, get_dataloader
from score2018 import Challenge2018Score
from visualizations import plot_confusion_matrix, plot_classes_distribution, sleep_staging_confusion




# sleep labels for SHHS and HMC [0, 1, 2, 3, 4, 5] == ['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake']

# for dataset in [Path(HMC), Path(snooze), Path(SHHS)]:

def main():

    predict_1()

def dataset_stats():
    HMC = 'K:\\combined_HMC'

    snooze = 'D:\\data\\snooze\\marco'

    SHHS = 'K:\\shhs\\polysomnography\\shh1_numpy'

    for dataset in [Path(SHHS)]:
        files = sorted(dataset.glob("*"))

        shhs_hmc_dicto = {0: 3,
                          1: 5,
                          -1: 0,
                          -2: 1,
                          -3: 2,
                          -4: 2,
                          4: 4}

        sleep_collection = [[], [], [], [], [], []]
        arousal_collection = []
        # counter = 0

        for file in tqdm(files, total=len(files)):
            # print(file.suffix, file.stem, file)
            if file.suffix == ".pkl":
                continue
            if file.name == "arousal_IID_windows.pt" or file.name == "sleep_IID_windows.pt":
                continue
            if file.name == "RECORDS":
                continue
            if file.is_dir:
                # print(file)
                # marco is shhs data in this case
                if dataset.name == "marco":
                    y = torch.load(*file.glob("*labels*"))
                    arousals = y[0]
                    # print(np.unique(arousals, return_counts=True))
                    if 1 in np.unique(arousals, return_counts=True)[0]:
                        arousal_count = np.unique(arousals, return_counts=True)[1][2]
                        arousal_collection.append(arousal_count / len(arousals))
                    else:
                        arousal_collection.append(0)
                    sleep = y[1]
                else:
                    y = np.load(*file.glob("*labels*"))
                    arousals = y[0]
                    arousal_collection.append(arousals.sum() / len(arousals))

                    sleep = y[1]
                    # print(np.unique(sleep_, return_counts=True))
                    sleep = np.asarray(list(map(shhs_hmc_dicto.get, sleep)))
                    # print(np.unique(sleep, return_counts=True)[0])


                index, sleep = np.asarray(np.unique(sleep, return_counts=True))
                # print(index)
                sleep_ = np.zeros(6)
                for j, i in enumerate(index):
                    sleep_[i] = sleep[j]

                for i in range(len(sleep_collection)):
                    sleep_collection[i].append(int(sleep_[i]/6000))

                # counter += 1
                # if counter == 5:
                #     break

                if dataset.name == "marco":
                    datum = "snooze"
                elif dataset.name == "shh1_numpy":
                    datum = "SHHS"
                else:
                    datum = "HMC"

                plot_descriptors(sleep_collection, arousal_collection, datum)


def plot_descriptors(sleep_collection, arousal_collection, datum):

    means = np.mean(sleep_collection, axis=1)
    print(datum, means)
    print(datum, " arousal percentage: ", np.mean(arousal_collection))

    labels = ['nonrem1\n{}'.format(means[0]), 'nonrem2\n{}'.format(means[1]),
              'nonrem3\n{}'.format(means[2]), 'rem\n{}'.format(means[3]),
              'undefined\n{}'.format(means[4]), 'wake\n{}'.format(means[5])]
    sleep_collection = np.asarray(sleep_collection).T
    df = pd.DataFrame(sleep_collection, columns=labels)
    # print(df.head())

    # for i, stage in enumerate(sleep_collection):
    # sns.barplot(sleep_collection)

    Save_folder = Path("figures/statistics/general_data_description")
    Save_folder.mkdir(parents=True, exist_ok=True)

    sns.boxplot(data=df)
    plt.gca().set_title("sleep stages " + datum)
    plt.gca().set_xlabel('')
    plt.gca().set_ylabel("minutes")
    plt.show()

    filename = "sleep stages " + datum + '.png'
    plt.savefig(Save_folder / filename)
    plt.close("all")
    sns.barplot(y=arousal_collection)
    plt.gca().set_title("Arousals " + datum + " " + str(np.mean(arousal_collection)))
    plt.gca().set_xlabel('Arousal distribution')
    plt.show()

    filename = "Arousals " + datum + '.png'
    plt.savefig(Save_folder / filename)
    plt.close("all")


def predict_1():

    data_name = "SHHS"
    data_folder = 'K:\\shhs\\polysomnography\\shh1_numpy'
    data_name = "snooze"
    data_folder = 'D:\\data\\snooze\\marco'
    # data_name = "HMC"
    # data_folder = 'K:\\combined_HMC'
    # data_name = "philips"
    # data_folder = "."


    Challenge2018Scorer = Challenge2018Score()

    dataloader_params = {'batch_size': 2,
                         'shuffle': False,
                         'num_workers': 4}

    model_name = "Deep_Sleep"

    train_set, validation_set = get_dataloader(data_folder, model_name, data_name, size="default")
    validation_generator = data.DataLoader(validation_set, **dataloader_params)
    traning_generator = data.DataLoader(train_set, **dataloader_params)

    dataloaders = {"val": validation_generator, "train": traning_generator}

    accuracies = np.empty(0)
    kappas = np.empty(0)

    sleep_collection = [[], [], [], [], [], []]
    arousal_collection = []

    # for phase in ["val", "train"]:
    count = 0
    for phase in ["val"]:
        for ID, __, annotations_arousal, annotations_sleep in dataloaders[phase]:
            with torch.set_grad_enabled(False):

                index, sleep = np.asarray(np.unique(annotations_sleep, return_counts=True))
                # print(index)
                sleep_ = np.zeros(6)
                for j, i in enumerate(index):
                    sleep_[i] = sleep[j]

                for i in range(len(sleep_collection)):
                    sleep_collection[i].append(int(sleep_[i] / 6000))

                if 1 in np.unique(annotations_arousal, return_counts=True)[0]:
                    arousal_count = np.unique(annotations_arousal, return_counts=True)[1][2]
                    # print(arousal_count, annotations_arousal.size(), len(annotations_arousal), annotations_arousal.shape)
                    # print(asasa)
                    arousal_collection.append((arousal_count / annotations_arousal.size()[-1])/dataloader_params['batch_size'])
                else:
                    arousal_collection.append(0)


                true_array_ = annotations_arousal.cpu().numpy().squeeze().astype(int)
                ones = np.ones(annotations_arousal.shape)

                do_auroc(Challenge2018Scorer, true_array_, ones, ID, "predict_one", 0)
                print("AUPRC, AUROC", Challenge2018Scorer.gross_auprc(), Challenge2018Scorer.gross_auroc())
                accuracies_, kappas_ = do_stats_arousal(true_array_, ones, ID)
                accuracies = np.append(accuracies, accuracies_)
                kappas = np.append(kappas, kappas_)

                print(count, " balanced accuracies, kappas", accuracies.mean(), kappas.mean())

                count +=1

                if count == 50:
                    break

    plot_descriptors(sleep_collection, arousal_collection, data_name)

def do_auroc(Challenge2018Scorer, true_array_, prediction_prob_arousal, ID, comment, epoch):

    # fpr, tpr = 0, 0
    # fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    # fig.suptitle('{}'.format(comment), fontsize=16)
    # lw = 2
    # ax[0].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # cmap = plt.cm.Paired(np.linspace(0, 1, len(true_array_)))

    for i, _ in enumerate(true_array_):
        true = true_array_[i]
        pred = prediction_prob_arousal[i]
        identity = ID[i]

        if true_array_.ndim == 1:
            true = true_array_
            pred = prediction_prob_arousal
            identity = None

        Challenge2018Scorer.score_record(true[true != 0] - 1, pred[true != 0], identity)

    #     # print(Challenge2018Scorer._record_auc)
    #
    #     fpr, tpr, _ = metrics.roc_curve(true[true != 0] - 1, pred[true != 0])
    #     prec, recall, _ = metrics.precision_recall_curve(true[true != 0] - 1, pred[true != 0], pos_label=1)
    #
    #     # print(fpr.shape, recall.shape, true[true != 0].shape)
    #
    #     roc_auc = metrics.auc(fpr, tpr)
    #     prc_auc = metrics.auc(recall, prec)
    #
    #     ax[0].plot(fpr, tpr, color=cmap[i],
    #              lw=lw, label='{} (area = {:0.2f})'.format(ID[i], roc_auc), alpha=0.7)
    #     ax[0].set_xlim([0.0, 1.0])
    #     ax[0].set_ylim([0.0, 1.05])
    #     ax[0].set_xlabel('False Positive Rate')
    #     ax[0].set_ylabel('True Positive Rate')
    #     ax[0].set_title('Receiver operating characteristic')
    #     ax[0].legend(loc="lower right")
    #
    #     ax[1].plot(recall, prec, color=cmap[i],
    #              lw=lw, label='{} (area = {:0.2f})'.format(ID[i], prc_auc), alpha=0.7)
    #     ax[1].set_xlim([0.0, 1.0])
    #     ax[1].set_ylim([0.0, 1.05])
    #     ax[1].set_xlabel('Recall')
    #     ax[1].set_ylabel('Precision')
    #     ax[1].set_title('Precision Recall curve')
    #     ax[1].legend(loc="upper right")
    #
    #     # plt.hold(True)
    #     if true_array_.ndim == 1:
    #         break
    # Save_folder = Path("figures/aurocs") / comment
    # Save_folder.mkdir(parents=True, exist_ok=True)
    #
    # filename = comment + str(epoch) + '.png'
    # plt.savefig(Save_folder / filename)
    # plt.clf()
    # plt.close("all")


def do_stats_arousal(true, pred, id):

    accuracies = []
    kappas = []

    if true.ndim == 1:
        return metrics.balanced_accuracy_score(true, pred), metrics.cohen_kappa_score(true, pred, labels=[0, 1, 2])

    for i, _ in enumerate(true):
        pred_record = pred[i][true[i] != 0]
        true_record = true[i][true[i] != 0]
        accuracies.append(metrics.balanced_accuracy_score(true_record, pred_record))
        kappas.append(metrics.cohen_kappa_score(true_record, pred_record, labels=[0, 1, 2]))

        # if kappas == None:
        #     kappas = 0
        # if kappas == np.nan():
        #     kappas = 0

    return accuracies, kappas


if __name__ == '__main__':
    main()