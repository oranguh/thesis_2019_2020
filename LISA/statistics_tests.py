import os
import pickle as pkl
import platform
from pathlib import Path
from statistics import mean, stdev
from math import sqrt

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
import itertools
from itertools import combinations
import statsmodels.api as sm
import statsmodels.stats.multitest as smt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import array_to_latex as a2l

def main():
    # combined_data()
    weights()
    # weights_to_other()
    # sleep_staging()
    # channel_stats()
    # cross_dataset()
    # different_models()
    # frequency()
    # future_predict()


def frequency():
    accuracies = sorted(Path("statistics/accuracies/frequency").glob("*"))
    kappas = sorted(Path("statistics/kappas/frequency").glob("*"))

    df_accuracies = pd.DataFrame()
    df_kappas = pd.DataFrame()

    df_accuracies = name_replace_frequency(df_accuracies, accuracies)
    df_kappas = name_replace_frequency(df_kappas, kappas)

    # print(df_accuracies.columns, df_kappas.columns)
    for col in df_accuracies.columns:
        if col[0] != col[2]:
            df_accuracies = df_accuracies.drop(col, axis=1)
            df_kappas = df_kappas.drop(col, axis=1)

    # print(df_accuracies.columns)
    names = ["from_data", "frequency", "to_data"]
    df_accuracies.columns = pd.MultiIndex.from_tuples(df_accuracies.columns, names=names)
    df_kappas.columns = pd.MultiIndex.from_tuples(df_kappas.columns, names=names)
    df_accuracies = df_accuracies.sort_index(axis=1, level=[0], ascending=[True])
    df_kappas = df_kappas.sort_index(axis=1, level=[0], ascending=[True])

    y_labels = ["Balanced accuracy", "Kappas"]

    title = 'Frequency sampling Deep sleep'

    for i, df in enumerate([df_accuracies, df_kappas]):
        # df = df.loc[:, (data_name, slice(None))]

        # sns.violinplot(data=df, bw=.1)
        plt.subplots(figsize=(16, 8))
        sns.boxplot(data=df)
        plt.gca().set_title(title)
        plt.gca().set_xlabel('')
        plt.gca().set_ylabel(y_labels[i])
        plt.gca().set_ylim([-0.1, 1.1])
        plt.xticks(rotation=0, horizontalalignment="center")
        labels_figure = []
        for text in plt.gca().get_xticklabels():
            # print(text.get_text())
            temp_text = text.get_text()
            temp_text = list(temp_text.strip("()").replace("'", "").split(", "))
            temp_text = tuple(temp_text)
            correct_text = "{}, \n {} to {} \nmean: {:.2f}".format(temp_text[1], temp_text[0], temp_text[2], df.loc[:, temp_text].mean())
            text.set_text(correct_text)
            labels_figure.append(text)
        plt.gca().set_xticklabels(labels_figure)

        Save_folder = Path("figures/statistics/frequency")
        Save_folder.mkdir(parents=True, exist_ok=True)

        filename = title + y_labels[i] + '.png'
        filename = filename.replace(" ", "_")
        plt.savefig(Save_folder / filename)

        plt.show()


def different_models():
    accuracies = sorted(Path("statistics/accuracies/convnet").glob("*"))
    kappas = sorted(Path("statistics/kappas/convnet").glob("*"))

    df_accuracies = pd.DataFrame()
    df_kappas = pd.DataFrame()

    df_accuracies = name_replace_convnet(df_accuracies, accuracies)
    df_kappas = name_replace_convnet(df_kappas, kappas)

    # for col in df_accuracies.columns:
    #     if col[0] == "combined+s":
    #         df_accuracies = df_accuracies.drop(col, axis=1)
    #         df_kappas = df_kappas.drop(col, axis=1)

    # print(df_accuracies.columns)
    names = ["model", "from_data", "to_data"]
    df_accuracies.columns = pd.MultiIndex.from_tuples(df_accuracies.columns, names=names)
    df_kappas.columns = pd.MultiIndex.from_tuples(df_kappas.columns, names=names)
    df_accuracies = df_accuracies.sort_index(axis=1, level=[0], ascending=[True])
    df_kappas = df_kappas.sort_index(axis=1, level=[0], ascending=[True])

    y_labels = ["Balanced accuracy", "Kappas"]
    # dataset_names = ["snooze", "SHHS", "philips", "HMC"]
    dataset_names = ["snooze", "SHHS"]

    for data_name in dataset_names:
        title = 'Various models ' + data_name
        for i, df in enumerate([df_accuracies, df_kappas]):
            df = df.loc[:, (slice(None), slice(None), data_name)]

            # sns.violinplot(data=df, bw=.1)
            plt.subplots(figsize=(16, 8))
            sns.boxplot(data=df)
            plt.gca().set_title(title)
            # plt.gca().set_xlabel('Effect of mixed datasets')
            plt.gca().set_ylabel(y_labels[i])
            plt.gca().set_ylim([-0.1, 1.1])
            plt.xticks(rotation=0, horizontalalignment="center")
            labels_figure = []
            for text in plt.gca().get_xticklabels():
                # print(text.get_text())
                temp_text = text.get_text()
                temp_text = list(temp_text.strip("()").replace("'", "").split(", "))
                temp_text = tuple(temp_text)
                correct_text = "{} \n {} to {} \nmean: {:.2f}".format(temp_text[0], temp_text[1], temp_text[2], df.loc[:, temp_text].mean())
                text.set_text(correct_text)
                labels_figure.append(text)

            plt.gca().set_xticklabels(labels_figure)

            Save_folder = Path("figures/statistics/different_models")
            Save_folder.mkdir(parents=True, exist_ok=True)

            filename = title + y_labels[i] + '.png'
            filename = filename.replace(" ", "_")
            plt.savefig(Save_folder / filename)

            plt.show()



def combined_data():
    accuracies = sorted(Path("statistics/accuracies/combined_data").glob("*"))
    kappas = sorted(Path("statistics/kappas/combined_data").glob("*"))

    df_accuracies = pd.DataFrame()
    df_kappas = pd.DataFrame()

    df_accuracies = name_replace_combined_data(df_accuracies, accuracies)
    df_kappas = name_replace_combined_data(df_kappas, kappas)

    # for col in df_accuracies.columns:
    #     if col[0] == "combined+s":
    #         df_accuracies = df_accuracies.drop(col, axis=1)
    #         df_kappas = df_kappas.drop(col, axis=1)

    # print(df_accuracies.columns)
    names = ["from_data", "to_data"]
    df_accuracies.columns = pd.MultiIndex.from_tuples(df_accuracies.columns, names=names)
    df_kappas.columns = pd.MultiIndex.from_tuples(df_kappas.columns, names=names)
    df_accuracies = df_accuracies.sort_index(axis=1, level=[1], ascending=[True])
    df_kappas = df_kappas.sort_index(axis=1, level=[1], ascending=[True])

    y_labels = ["Balanced accuracy", "Kappas"]
    title = 'Inter dataset generalizability'

    for i, df in enumerate([df_accuracies, df_kappas]):
        # df = df.loc[:, (data_name, slice(None))]

        # sns.violinplot(data=df, bw=.1)
        plt.subplots(figsize=(16, 8))
        sns.boxplot(data=df)
        plt.gca().set_title(title)
        plt.gca().set_xlabel('Effect of mixed datasets')
        plt.gca().set_ylabel(y_labels[i])
        plt.gca().set_ylim([-0.1, 1.1])
        plt.xticks(rotation=0, horizontalalignment="center")
        labels_figure = []
        for text in plt.gca().get_xticklabels():
            # print(text.get_text())
            temp_text = text.get_text()
            temp_text = list(temp_text.strip("()").replace("'", "").split(", "))
            temp_text = tuple(temp_text)
            correct_text = "{} to {} \nmean: {:.2f}".format(temp_text[0], temp_text[1], df.loc[:, temp_text].mean())
            text.set_text(correct_text)
            labels_figure.append(text)
        plt.gca().set_xticklabels(labels_figure)

        Save_folder = Path("figures/statistics/combined_data")
        Save_folder.mkdir(parents=True, exist_ok=True)

        filename = title + y_labels[i] + '.png'
        filename = filename.replace(" ", "_")
        plt.savefig(Save_folder / filename)

        plt.show()

def weights_to_other():

    accuracies = sorted(Path("statistics/accuracies/weights").glob("*weights*"))
    kappas = sorted(Path("statistics/kappas/weights").glob("*weights*"))
    AUPRC = sorted(Path("statistics/AUPRC/weights").glob("*weights*"))

    df_accuracies = pd.DataFrame()
    df_kappas = pd.DataFrame()
    df_AUPRC = pd.DataFrame()

    df_accuracies = name_replace_weights(df_accuracies, accuracies)
    df_kappas = name_replace_weights(df_kappas, kappas)
    df_AUPRC = name_replace_weights(df_AUPRC, AUPRC)

    names = ["from_data", "weight", "to_data"]
    df_accuracies.columns = pd.MultiIndex.from_tuples(df_accuracies.columns, names=names)
    df_kappas.columns = pd.MultiIndex.from_tuples(df_kappas.columns, names=names)
    df_AUPRC.columns = pd.MultiIndex.from_tuples(df_AUPRC.columns, names=names)

    df_accuracies = df_accuracies.sort_index(axis=1, level=[1], ascending=[True])
    df_kappas = df_kappas.sort_index(axis=1, level=[1], ascending=[True])
    df_AUPRC = df_AUPRC.sort_index(axis=1, level=[1], ascending=[True])

    y_labels = ["Balanced accuracy", "Kappas"]
    dataset_names = ["snooze", "SHHS", "philips", "HMC"]


    for data_name in dataset_names:
        for i, df in enumerate([df_accuracies, df_kappas]):
            title = 'Weight ratio on {}'.format(data_name)
            # print(df.columns)
            df = df.loc[:, (slice(None), 0.01, data_name)]
            # print(df.columns)
            # # print(*df.values.transpose())
            # print(stats.levene(*df.values.transpose()))
            # print([stats.normaltest(x) for x in df.values.transpose()])
            # print(stats.mannwhitneyu(df.values.transpose()[0], df.values.transpose()[1]))
            # print(asas)
            # sns.violinplot(data=df, bw=.1)
            plt.subplots(figsize=(16, 8))
            sns.boxplot(data=df)
            plt.gca().set_title(title)
            plt.gca().set_xlabel('Non-Arousal importance')
            plt.gca().set_ylabel(y_labels[i])
            plt.gca().set_ylim([-0.1, 1.1])
            plt.xticks(rotation=0, horizontalalignment="right")
            labels_figure = []
            # for text in plt.gca().get_xticklabels():
            #     # print(text.get_text())
            #     temp_text = text.get_text()
            #     temp_text = list(temp_text.strip("()").replace("'", "").split(", "))
            #     temp_text[1] = float(temp_text[1])
            #     temp_text = tuple(temp_text)
            #     correct_text = "weights: {}\nmean: {:.2f}".format(temp_text[1], df.loc[:, temp_text].mean())
            #     text.set_text(correct_text)
            #     labels_figure.append(text)
            # plt.gca().set_xticklabels(labels_figure)

            Save_folder = Path("figures/statistics/to_other_datasets")
            Save_folder.mkdir(parents=True, exist_ok=True)

            # filename = title + y_labels[i] + '.png'
            # filename = filename.replace(" ", "_")
            # plt.savefig(Save_folder / filename)

            plt.show()


def weights():

    accuracies = sorted(Path("statistics/accuracies/weights").glob("*weights*"))
    kappas = sorted(Path("statistics/kappas/weights").glob("*weights*"))
    AUPRC = sorted(Path("statistics/AUPRC/weights").glob("*weights*"))

    df_accuracies = pd.DataFrame()
    df_kappas = pd.DataFrame()
    df_AUPRC = pd.DataFrame()

    df_accuracies = name_replace_weights(df_accuracies, accuracies)
    df_kappas = name_replace_weights(df_kappas, kappas)
    df_AUPRC = name_replace_weights(df_AUPRC, AUPRC)

    names = ["from_data", "weight", "to_data"]
    df_accuracies.columns = pd.MultiIndex.from_tuples(df_accuracies.columns, names=names)
    df_kappas.columns = pd.MultiIndex.from_tuples(df_kappas.columns, names=names)
    df_AUPRC.columns = pd.MultiIndex.from_tuples(df_AUPRC.columns, names=names)

    df_accuracies = df_accuracies.sort_index(axis=1, level=[1], ascending=[True])
    df_kappas = df_kappas.sort_index(axis=1, level=[1], ascending=[True])
    df_AUPRC = df_AUPRC.sort_index(axis=1, level=[1], ascending=[True])

    y_labels = ["AUPRC", "Balanced accuracy", "Kappas"]
    dataset_names = ["snooze", "HMC", "SHHS", "philips"]

    Save_folder = Path("figures/statistics/weights")
    Save_folder.mkdir(parents=True, exist_ok=True)

    weights_correlates = []
    for data_name in dataset_names:
        weights_data = []
        fig_line, ax_line = plt.subplots(figsize=(16, 8))
        palette = itertools.cycle(sns.color_palette())

        for i, df in enumerate([df_AUPRC, df_accuracies, df_kappas]):
            title = 'Weight ratio on {}'.format(data_name)

            df = df.loc[:, (data_name, slice(None), data_name)]

            make_table(df)
            # print(df.columns)
            # # print(*df.values.transpose())
            # print(stats.levene(*df.values.transpose()))
            # print([stats.normaltest(x) for x in df.values.transpose()])
            # print(stats.mannwhitneyu(df.values.transpose()[0], df.values.transpose()[1]))
            # print(asas)
            # sns.violinplot(data=df, bw=.1)
            fig, ax = plt.subplots(figsize=(16, 8))
            sns.boxplot(data=df)
            # sns.lineplot(data=df.mean())
            ax.set_title(title)
            ax.set_xlabel('Non-Arousal importance')
            ax.set_ylabel(y_labels[i])
            ax.set_ylim([-0.1, 1.1])
            # ax.set_xticks(rotation=0, horizontalalignment="right")
            labels_figure = []
            for text in ax.get_xticklabels():
                # print(text.get_text())
                temp_text = text.get_text()
                temp_text = list(temp_text.strip("()").replace("'", "").split(", "))
                temp_text[1] = float(temp_text[1])
                temp_text = tuple(temp_text)
                correct_text = "weights: {}\nmean: {:.3f}".format(temp_text[1], df.loc[:, temp_text].mean())
                text.set_text(correct_text)
                labels_figure.append(text)
            ax.set_xticklabels(labels_figure)

            filename = title + y_labels[i] + '.png'
            filename = filename.replace(" ", "_")
            fig.savefig(Save_folder / filename)

            fig_line, ax_line, best_weight = plot_line(df, y_labels[i], fig_line, ax_line, palette)

            weights_data.append(best_weight)
        weights_correlates.append(weights_data)
        ax_line.set_title(data_name)
        filename = data_name + "_regressions.png"
        fig_line.savefig(Save_folder / filename)
            # plt.show()

    plot_weight_correlate(weights_correlates, y_labels, dataset_names, [0.078, 0.0266, 0.043, 0.004], Save_folder)

def channel_stats():

    accuracies = sorted(Path("statistics/accuracies/channel").glob("*"))
    kappas = sorted(Path("statistics/kappas/channel").glob("*"))

    df_accuracies = pd.DataFrame()
    df_kappas = pd.DataFrame()
    df_accuracies = name_replace_channels(df_accuracies, accuracies)
    df_kappas = name_replace_channels(df_kappas, kappas)

    for col in df_accuracies.columns:
        if col[0] != col[2]:
            df_accuracies = df_accuracies.drop(col, axis=1)
            df_kappas = df_kappas.drop(col, axis=1)
        elif col[1] == col[3]:
            df_accuracies = df_accuracies.drop(col, axis=1)
            df_kappas = df_kappas.drop(col, axis=1)
        elif col[1] != "random":
            df_accuracies = df_accuracies.drop(col, axis=1)
            df_kappas = df_kappas.drop(col, axis=1)
        else:
            pass

    names = ["from_data", "from_channel", "to_data", "to_channel"]
    df_accuracies.columns = pd.MultiIndex.from_tuples(df_accuracies.columns, names=names)
    df_kappas.columns = pd.MultiIndex.from_tuples(df_kappas.columns, names=names)

    y_labels = ["Balanced accuracy", "Kappas"]
    dataset_names = ["snooze", "SHHS"]

    for data_name in dataset_names:
        for i, df in enumerate([df_accuracies, df_kappas]):
            title = 'Channels on {}'.format(data_name)
            df = df.loc[:, (data_name, slice(None), slice(None), slice(None))]

            make_table(df)

            # print(a2l.to_ltx(smt.multipletests(b, method='bonferroni')[1], frmt = '{:6.2f}', arraytype = 'array', mathform=True))
            # sns.violinplot(data=df, bw=.1)
            plt.subplots(figsize=(16, 8))
            sns.boxplot(data=df)
            plt.gca().set_title(title)
            plt.gca().set_xlabel('Channel Train to Test')
            plt.gca().set_ylabel(y_labels[i])
            plt.gca().set_ylim([-0.1, 1.1])
            plt.xticks(rotation=25, horizontalalignment="right")
            labels_figure = []
            for text in plt.gca().get_xticklabels():
                # print(text.get_text())
                temp_text = text.get_text()
                temp_text = tuple(temp_text.strip("()").replace("'", "").split(", "))
                correct_text = "{} to {} \nmean: {:.2f}".format(temp_text[1], temp_text[3], df.loc[:, temp_text].mean())
                text.set_text(correct_text)
                labels_figure.append(text)
            plt.gca().set_xticklabels(labels_figure)

            Save_folder = Path("figures/statistics/channel_stats")
            Save_folder.mkdir(parents=True, exist_ok=True)

            filename = title + y_labels[i] + '.png'
            filename = filename.replace(" ", "_")
            plt.savefig(Save_folder / filename)#, bbox_inches='tight',pad_inches = 0)

            plt.show()


def cross_dataset():
    accuracies = sorted(Path("statistics/accuracies/channel").glob("*"))
    kappas = sorted(Path("statistics/kappas/channel").glob("*"))

    df_accuracies = pd.DataFrame()
    df_kappas = pd.DataFrame()

    df_accuracies = name_replace_channels(df_accuracies, accuracies)
    df_kappas = name_replace_channels(df_kappas, kappas)

    for col in df_accuracies.columns:
        if col[0] == col[2]:
            df_accuracies = df_accuracies.drop(col, axis=1)
            df_kappas = df_kappas.drop(col, axis=1)

    names = ["from_data", "from_channel", "to_data", "to_channel"]
    df_accuracies.columns = pd.MultiIndex.from_tuples(df_accuracies.columns, names=names)
    df_kappas.columns = pd.MultiIndex.from_tuples(df_kappas.columns, names=names)

    y_labels = ["Balanced accuracy", "Kappas"]
    dataset_names = ["snooze", "SHHS"]

    for data_name in dataset_names:
        for i, df in enumerate([df_accuracies, df_kappas]):
            title = 'Data generalization from {} (inter data)'.format(data_name)
            # df = df.loc[:, ("snooze", slice(None), slice(None), slice(None))]
            df = df.loc[:, (data_name, slice(None), slice(None), slice(None))]

            # sns.violinplot(data=df, bw=.1)
            plt.subplots(figsize=(16, 8))
            sns.boxplot(data=df)

            plt.gca().set_title(title)
            plt.gca().set_xlabel('Channel Train to Test')
            plt.gca().set_ylabel(y_labels[i])
            plt.xticks(rotation=15, horizontalalignment="right")
            plt.gca().set_ylim([-0.1, 1.1])
            labels_figure = []
            for text in plt.gca().get_xticklabels():
                temp_text = text.get_text()
                temp_text = tuple(temp_text.strip("()").replace("'", "").split(", "))
                correct_text = "{} to {} \nmean: {:.2f}".format(temp_text[1], temp_text[3], df.loc[:, temp_text].mean())
                text.set_text(correct_text)
                labels_figure.append(text)
            plt.gca().set_xticklabels(labels_figure)

            Save_folder = Path("figures/statistics/cross_dataset")
            Save_folder.mkdir(parents=True, exist_ok=True)

            filename = title + y_labels[i] + '.png'
            filename = filename.replace(" ", "_")
            plt.savefig(Save_folder / filename)

            plt.show()


def sleep_staging():
    accuracies = sorted(Path("statistics/accuracies/sleep_staging").glob("*"))
    kappas = sorted(Path("statistics/kappas/sleep_staging").glob("*"))

    df_accuracies = pd.DataFrame()
    df_kappas = pd.DataFrame()

    df_accuracies = name_replace_sleep_stage(df_accuracies, accuracies)
    df_kappas = name_replace_sleep_stage(df_kappas, kappas)

    for col in df_accuracies.columns:
        if col[0] != col[2]:
            df_accuracies = df_accuracies.drop(col, axis=1)
            df_kappas = df_kappas.drop(col, axis=1)

    names = ["from_data", "weight", "to_data"]
    df_accuracies.columns = pd.MultiIndex.from_tuples(df_accuracies.columns, names=names)
    df_kappas.columns = pd.MultiIndex.from_tuples(df_kappas.columns, names=names)
    df_accuracies = df_accuracies.sort_index(axis=1, level=[1], ascending=[True])
    df_kappas = df_kappas.sort_index(axis=1, level=[1], ascending=[True])

    y_labels = ["Balanced accuracy", "Kappas"]
    dataset_names = ["snooze", "SHHS"]
    # dataset_names = ["SHHS"]

    for data_name in dataset_names:
        for i, df in enumerate([df_accuracies, df_kappas]):
            title = 'Sleep staging effect on {} (intra data)'.format(data_name)
            df = df.loc[:, (data_name, slice(None), data_name)]

            # sns.violinplot(data=df, bw=.1)
            plt.subplots(figsize=(16, 8))
            sns.boxplot(data=df)
            plt.gca().set_title(title)
            plt.gca().set_xlabel('Sleep staging importance')
            plt.gca().set_ylabel(y_labels[i])
            plt.gca().set_ylim([-0.1, 1.1])
            plt.xticks(rotation=0, horizontalalignment="right")
            labels_figure = []
            for text in plt.gca().get_xticklabels():
                # print(text.get_text())
                temp_text = text.get_text()
                temp_text = list(temp_text.strip("()").replace("'", "").split(", "))
                temp_text[1] = float(temp_text[1])
                temp_text = tuple(temp_text)
                correct_text = "weights: {}\nmean: {:.2f}".format(temp_text[1], df.loc[:, temp_text].mean())
                text.set_text(correct_text)
                labels_figure.append(text)
            plt.gca().set_xticklabels(labels_figure)

            Save_folder = Path("figures/statistics/sleep_staging")
            Save_folder.mkdir(parents=True, exist_ok=True)

            filename = title + y_labels[i] + '.png'
            filename = filename.replace(" ", "_")
            plt.savefig(Save_folder / filename)

            plt.show()


def name_replace_frequency(df, paths):
    for i, _ in enumerate(paths):
        data = np.load(paths[i])

        name = paths[i].name
        name = name.split("Banana")[-1]
        name = name.split("ghildes")[-1]
        name = name.replace(".npy", "")
        name = name.replace("Deep_Sleep_SHHS", "SHHS").replace("Deep_Sleep_snooze", "snooze").replace("Deep_Sleep_philips", "philips_")
        name = name.replace("Deep_Sleep_combined_combined_dataset", "combined")
        name = name.replace("_to_", "_")

        df_name = tuple(name.split("_"))
        df.insert(0, df_name, pd.Series(data))

    return df


def name_replace_convnet(df, paths):
    for i, _ in enumerate(paths):
        data = np.load(paths[i])

        name = paths[i].name
        name = name.split("Banana")[-1]
        name = name.split("ghildes")[-1]
        name = name.replace(".npy", "")
        name = name.replace("SHHS", "SHHS").replace("snooze", "snooze").replace("philips", "philips")
        name = name.replace("philips", "philips")

        name = name.replace("ConvNet_IID_", "ConvNet_").replace("Deep_Sleep_", "Deep-Sleep_").replace("Howe_Patterson", "Patterson-LSTM")

        name = name.replace("_to_", "")

        df_name = tuple(name.split("_"))
        # print(df_name, name, paths[i].name)
        df.insert(0, df_name, pd.Series(data))

    return df


def name_replace_combined_data(df, paths):
    for i, _ in enumerate(paths):
        data = np.load(paths[i])

        name = paths[i].name
        name = name.split("Banana")[-1]
        name = name.split("ghildes")[-1]
        name = name.replace(".npy", "")
        name = name.replace("Deep_Sleep_SHHS", "SHHS_").replace("Deep_Sleep_snooze", "snooze_").replace("Deep_Sleep_philips", "philips")
        name = name.replace("Deep_Sleep_combined_combined_dataset", "combined_").replace("Deep_Sleep_combined_sleep_staging", "combined+s_")
        name = name.replace("_to_", "")

        df_name = tuple(name.split("_"))
        df.insert(0, df_name, pd.Series(data))

    return df


def name_replace_sleep_stage(df, paths):
    for i, _ in enumerate(paths):
        data = np.load(paths[i])

        name = paths[i].name
        name = name.split("Banana")[-1]
        name = name.split("ghildes")[-1]
        name = name.replace(".npy", "")
        name = name.replace("Deep_Sleep_SHHS", "SHHS_").replace("Deep_Sleep_snooze", "snooze_").replace("Deep_Sleep_philips", "philips")
        name = name.replace("sleep_", "")
        name = name.replace("_to_", "_")
        #TODO this is wrong
        name = name.replace("1_10", "0.1").replace("1_5", "0.2").replace("1_2", "0.5").replace("1_1", "1").replace("1_0", "0")

        df_name = list(name.split("_"))
        df_name[1] = float(df_name[1])
        df_name = tuple(df_name)
        df.insert(0, df_name, pd.Series(data))

    return df


def name_replace_weights(df, paths):
    for i, _ in enumerate(paths):
        data = np.load(paths[i])

        name = paths[i].name
        name = name.split("Banana")[-1]
        name = name.split("ghildes")[-1]
        name = name.replace(".npy", "")
        name = name.replace("Deep_Sleep_SHHS", "SHHS_")\
            .replace("Deep_Sleep_snooze", "snooze_")\
            .replace("Deep_Sleep_philips", "philips")\
            .replace("Deep_Sleep_HMC", "HMC")
        name = name.replace("weights_", "")
        name = name.replace("_to_", "_")
        #TODO this is wrong
        name = name \
            .replace("1_1000", "0.001") \
            .replace("1_200", "0.005")\
            .replace("1_100", "0.01") \
            .replace("1_50", "0.02")\
            .replace("1_5", "0.2")\
            .replace("1_20", "0.05")\
            .replace("1_10", "0.1")\
            .replace("1_1", "1")

        df_name = list(name.split("_"))
        # print(df_name)
        df_name[1] = float(df_name[1])
        df_name = tuple(df_name)
        # print(data.shape)
        df.insert(0, df_name, pd.Series(data))

    return df

def name_replace_channels(df, paths):
    for i, _ in enumerate(paths):
        data = np.load(paths[i])

        name = paths[i].name
        name = name.split("Banana")[-1]
        name = name.split("ghildes")[-1]
        name = name.replace(".npy", "")
        name = name.replace("Deep_Sleep_SHHS", "SHHS_").replace("Deep_Sleep_snooze", "snooze_").replace("Deep_Sleep_philips", "philips")
        name = name.replace("_to_", "_")
        name = name.replace("random_channel", "random")
        name = name.replace("C3_M2", "C3-M2").replace("C4_M1", "C4-M1").replace("F3_M2", "F3-M2").replace("F4_M1", "F4-M1")
        name = name.replace("C3_A2", "C3-A2").replace("C4_A1", "C4-A1")
        # print(name.split("_"))

        df_name = tuple(name.split("_"))
        # print(df_name)
        df.insert(0, df_name, pd.Series(data))

    return df

def make_table(df):
    print(df.columns)
    # print(stats.levene(*df.values.transpose()))
    # print([stats.normaltest(x) for x in df.values.transpose()])
    # print(stats.mannwhitneyu(df.values.transpose()[0], df.values.transpose()[1]))

    perm = list(combinations(list(range(len(df.values.transpose()))), 2))
    b = []
    latex_df = pd.DataFrame(columns=['Effect size', 'p value'])
    for j in perm:
        # print(j)
        df.dropna(inplace=True)
        sample_1 = df.values.transpose()[j[0]]
        sample_2 = df.values.transpose()[j[1]]
        a = stats.mannwhitneyu(sample_1, sample_2)
        # latex_df = latex_df.append([j, a[0], a[1]], ignore_index=True)
        b.append(a[1])
        cohens_d = (mean(sample_1) - mean(sample_2)) / (sqrt((stdev(sample_1) ** 2 + stdev(sample_2) ** 2) / 2))

        latex_df = latex_df.append({'Effect size': abs(cohens_d), 'p value': a[1]}, ignore_index=True)
        # print(a)
        # print()
    # print(perm)
    # print(smt.multipletests(b, method='bonferroni'))

    latex_df.insert(2, "p corrected", pd.Series(smt.multipletests(b, method='bonferroni')[1]))
    latex_df.insert(3, "Significant", [str(x) for x in pd.Series(smt.multipletests(b, method='bonferroni')[0])] )
    latex_df.index = [str(x) for x in perm]
    print("data shape", df.values.transpose().shape)
    print(latex_df)
    print(a2l.to_ltx(latex_df, arraytype='tabular'))

    latex_df = pd.DataFrame(columns=['Name', 'N', 'mean', 'std'])
    for i in range(len(df.values.transpose())):

        name = ''.join([str(x) for x in df.columns[i]])
        latex_df = latex_df.append({'Name': name, 'N': len(df.values.transpose()[i]), 'mean': mean(df.values.transpose()[i]),
                                    'std': stdev(df.values.transpose()[i])}, ignore_index=True)

        print({'Name': name, 'N': len(df.values.transpose()[i]), 'mean': mean(df.values.transpose()[i]),
               'std': stdev(df.values.transpose()[i])})
    print(df.describe())
    print(a2l.to_ltx(latex_df, arraytype='tabular'))

    # print(asas)
def plot_line(df, y_label, fig, ax, palette):

    color = next(palette)
    # plt.close("all")
    # plt.cla()
    bins = len(df.values.transpose())

    df_plot = df.unstack().droplevel(0).droplevel(1).unstack(0).stack(level=['weight']).reset_index()
    df_plot.weight = np.log(df_plot["weight"])
    df_plot.rename(columns={"level_0":"random", "weight":"log weight", 0:y_label}, inplace=True)

    sns.regplot(x="log weight", y=y_label, data=df_plot, scatter_kws={"alpha": 0.3}, order=2, ax=ax, label=y_label, x_bins=bins, color=color)
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlim([df_plot["log weight"].min() - 0.1, df_plot["log weight"].max() + 0.1])
    ax.set_ylabel("Score")

    grouped = df_plot.groupby("log weight").mean()
    # print(grouped[y_label].max())
    max_value = grouped.loc[grouped[y_label] == grouped[y_label].max()].index[0]
    # ax.axvline(max_value, 0, grouped[y_label].max(), color=color, alpha=0.5) # Don't you just love pandas
    # sns.swarmplot(x="weight", y="value", data=df_plot)
    legend = ax.legend(loc="best")
    return fig, ax, max_value


    # plt.show()
    plt.close("all")
    plt.cla()

    # df_plot = df_plot[df_plot["log weight"] != 0]
    if False:
        x = df_plot["log weight"]
        y = df_plot[y_label] - np.mean(df_plot[y_label])
        model = sm.OLS(y, x).fit()
        # predictions = model.predict(X)  # make the predictions by the model

        # Print out the statistics
        print(model.summary())
        prstd, iv_l, iv_u = wls_prediction_std(model)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y, 'o', label="data", alpha=0.5)
        ax.plot(x, model.fittedvalues, 'r--.', label="OLS")
        # ax.plot(x, iv_u, 'r--')
        # ax.plot(x, iv_l, 'r--')
        plt.show()

def plot_weight_correlate(weights_correlates, y_labels, dataset_names, arousal_percentages, Save_folder):
    plt.close("all")
    plt.cla()

    weights_correlates = np.asarray(weights_correlates)
    weights_correlates = weights_correlates.transpose()
    weights_correlates = np.exp(weights_correlates)

    df = pd.DataFrame(data=weights_correlates, index=y_labels, columns=dataset_names)
    df = df.append(pd.DataFrame(data=np.asarray(arousal_percentages)[np.newaxis, :], index=["Percentage arousal"], columns=dataset_names))
    print(a2l.to_ltx(df, frmt='{:1.3f}', arraytype='tabular'))

    for i, label in enumerate(y_labels):
        fig, ax = plt.subplots(figsize=(16, 8))
        # sns.regplot(x=arousal_percentages, y=weights_correlates[i], scatter_kws={"alpha": 1}, label=label)
        sns.scatterplot(x=arousal_percentages, y=weights_correlates[i], ax=ax)
        ax.set_xlabel("Arousal Percentage")
        ax.set_ylabel("Optimal weight setting")
        # plt.gca().plot(plt.gca().get_xlim(), plt.gca().get_ylim(), ls="--", c=".3", color="red")
        add_identity(ax, color='r', ls='--')
        ax.legend(loc="best")
        ax.set_title(label + " on datasets")

        for j, txt in enumerate(dataset_names):
            ax.annotate(txt, (arousal_percentages[j], weights_correlates[i][j]))

        filename = label + "_correlates"
        filename = filename.replace(" ", "_")
        fig.savefig(Save_folder / filename)
        # plt.show()
        plt.close("all")
        plt.cla()

def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs, label="identity")
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

if __name__ == '__main__':
    main()
