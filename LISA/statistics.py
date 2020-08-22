import os
import pickle as pkl
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
import matplotlib.pyplot as plt


def main():

    stat_data_SHHS = sorted(Path("statistics").glob("*SHHS*weights*SHHS*"))
    stat_data_snooze = sorted(Path("statistics").glob("*weights*SHHS*"))

    print(stat_data_SHHS)

    fig1, ax1 = plt.subplots()

    for i, _ in enumerate(stat_data_SHHS):
        a = np.load(stat_data_SHHS[i])
        # print(a)
        ax1.boxplot(a, positions = [i])

    ax1.set_title('Weights and Balanced Accuracy SHHS')
    ax1.xaxis.set_ticklabels(["2%", "5%", "10%", "20%", "100%"])
    ax1.set_xlabel('Relative Non-Arousal-weight')
    ax1.set_ylabel('Balanced Accuracies')
    plt.show()


    # for i, _ in enumerate(stat_data_SHHS):
    #     print(stat_data_SHHS[i], stat_data_snooze[i])





if __name__ == '__main__':
    main()