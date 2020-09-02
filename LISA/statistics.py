import os
import pickle as pkl
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    arousal_stat_data_SHHS = sorted(Path("statistics/accuracies").glob("*SHHS*weights*SHHS*"))
    arousal_stat_data_SHHS = sorted(Path("statistics/kappas").glob("*SHHS*weights*SHHS*"))
    stat_data_snooze = sorted(Path("statistics").glob("*weights*SHHS*"))

    print(arousal_stat_data_SHHS)

    print(arousal_stat_data_SHHS[0].name.split("Banana")[-1])
    fig1, ax1 = plt.subplots()

    df = pd.DataFrame()

    for i, _ in enumerate(arousal_stat_data_SHHS):
        a = np.load(arousal_stat_data_SHHS[i])
        # print(a)
        # ax1.boxplot(a, positions = [i])
        name = arousal_stat_data_SHHS[i].name.split("Banana")[-1].replace(".npy", "").replace("Deep_Sleep_SHHS", "").replace("_to_SHHS", "").replace("weights_1_", "")
        df.insert(0, 1/float(name), a)

    # ax1.set_title('Weights and Balanced Accuracy SHHS')
    # ax1.xaxis.set_ticklabels(["2%", "5%", "10%", "20%", "100%"])
    # ax1.set_xlabel('Relative Non-Arousal-weight')
    # ax1.set_ylabel('Kappas')
    # plt.show()

    print(df.head())
    df.sort_index(axis=1, inplace=True)
    df.boxplot()
    plt.gca().set_title('Weights and Kappas SHHS to SHHS')
    plt.gca().set_xlabel('Relative Non-Arousal-weight')
    plt.gca().set_ylabel('Kappas')
    plt.show()
    # for i, _ in enumerate(stat_data_SHHS):
    #     print(stat_data_SHHS[i], stat_data_snooze[i])


def channel_stats():


    accuracies = sorted(Path("statistics/accuracies/channel").glob("*"))
    kappas = sorted(Path("statistics/kappas/channel").glob("*"))

    print(accuracies)
    df = pd.DataFrame()

    for i, _ in enumerate(accuracies):
        data = np.load(accuracies[i])

        name = accuracies[i].name.split("Banana")[-1].replace(".npy", "")
        name = name.replace("Deep_Sleep_SHHS", "SHHS_").replace("Deep_Sleep_snooze", "snooze_")
        name = name.replace("_to_", "_")
        name = name.replace("random_channel", "random")
        name = name.replace("C3_M2", "C3-M2").replace("C4_M1", "C4-M1").replace("F3_M2", "F3-M2").replace("F4_M1", "F4-M1")
        name = name.replace("C3_A2", "C3-A2").replace("C4_A1", "C4-A1")
        # print(name.split("_"))

        df_name = tuple(name.split("_"))
        # print(df_name)
        df.insert(0, df_name, data)

    # print(df.head())
    names = ["from_data", "from_channel", "to_data", "to_channel"]
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=names)
    # print(df.head())

    # print(df["SHHS"]["C4-A1"]["snooze"]["F4-M1"])
    # print(df.loc[:, (slice("SHHS"), slice(None), "SHHS", slice(None))])


    a = df.loc[:, ("snooze", slice(None), "snooze", slice(None))]
    # a = df.loc[:, ("SHHS", slice(None), "SHHS", slice(None))]
    # axes = a.boxplot(return_type="axes")

    sns.violinplot(data=a, bw=.1)

    plt.gca().set_title('Channels on snooze data')
    plt.gca().set_xlabel('Channel Train to Test')
    plt.gca().set_ylabel('Kappas')
    plt.xticks(rotation=45)
    labels_figure = []
    for text in plt.gca().get_xticklabels():
        # print(text.get_text())
        temp_text = text.get_text()
        temp_text = tuple(temp_text.strip("()").replace("'", "").split(","))
        correct_text = "{} to {}".format(temp_text[1], temp_text[3])
        text.set_text(correct_text)
        labels_figure.append(text)
    plt.gca().set_xticklabels(labels_figure)

    plt.show()




if __name__ == '__main__':
    channel_stats()
    # main()