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
    # combined_data()
    # weights()
    # sleep_staging()
    # channel_stats()
    # cross_dataset()
    different_models()
    frequency()
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

        Save_folder = Path("figures/statistics/frequency") / title + y_labels[i]
        Save_folder.mkdir(parents=True, exist_ok=True)

        filename = title + '.png'
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
    names = ["from_data", "to_data"]
    df_accuracies.columns = pd.MultiIndex.from_tuples(df_accuracies.columns, names=names)
    df_kappas.columns = pd.MultiIndex.from_tuples(df_kappas.columns, names=names)
    df_accuracies = df_accuracies.sort_index(axis=1, level=[1], ascending=[True])
    df_kappas = df_kappas.sort_index(axis=1, level=[1], ascending=[True])

    y_labels = ["Balanced accuracy", "Kappas"]
    title = 'Convnet results'

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

        Save_folder = Path("figures/statistics/different_models")
        Save_folder.mkdir(parents=True, exist_ok=True)

        filename = title + y_labels[i] + '.png'
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
        plt.savefig(Save_folder / filename)

        plt.show()

def weights():

    accuracies = sorted(Path("statistics/accuracies/weights").glob("*weights*"))
    kappas = sorted(Path("statistics/kappas/weights").glob("*weights*"))

    df_accuracies = pd.DataFrame()
    df_kappas = pd.DataFrame()

    df_accuracies = name_replace_weights(df_accuracies, accuracies)
    df_kappas = name_replace_weights(df_kappas, kappas)

    names = ["from_data", "weight", "to_data"]
    df_accuracies.columns = pd.MultiIndex.from_tuples(df_accuracies.columns, names=names)
    df_kappas.columns = pd.MultiIndex.from_tuples(df_kappas.columns, names=names)
    df_accuracies = df_accuracies.sort_index(axis=1, level=[1], ascending=[True])
    df_kappas = df_kappas.sort_index(axis=1, level=[1], ascending=[True])

    y_labels = ["Balanced accuracy", "Kappas"]
    dataset_names = ["snooze", "SHHS", "philips", "HMC"]

    for data_name in dataset_names:
        for i, df in enumerate([df_accuracies, df_kappas]):
            title = 'Weight ratio on {}'.format(data_name)

            df = df.loc[:, (data_name, slice(None), data_name, slice(None))]

            # sns.violinplot(data=df, bw=.1)
            plt.subplots(figsize=(16, 8))
            sns.boxplot(data=df)
            plt.gca().set_title(title)
            plt.gca().set_xlabel('Non-Arousal importance')
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

            Save_folder = Path("figures/statistics/weights")
            Save_folder.mkdir(parents=True, exist_ok=True)

            filename = title + y_labels[i] + '.png'
            plt.savefig(Save_folder / filename)

            plt.show()


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

    names = ["from_data", "from_channel", "to_data", "to_channel"]
    df_accuracies.columns = pd.MultiIndex.from_tuples(df_accuracies.columns, names=names)
    df_kappas.columns = pd.MultiIndex.from_tuples(df_kappas.columns, names=names)

    y_labels = ["Balanced accuracy", "Kappas"]
    dataset_names = ["snooze", "SHHS"]

    for data_name in dataset_names:
        for i, df in enumerate([df_accuracies, df_kappas]):
            title = 'Channels on {} (intra data)'.format(data_name)
            df = df.loc[:, (data_name, slice(None), slice(None), slice(None))]

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
            plt.savefig(Save_folder / filename)

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
            plt.savefig(Save_folder / filename)

            plt.show()


def name_replace_frequency(df, paths):
    for i, _ in enumerate(paths):
        data = np.load(paths[i])

        name = paths[i].name
        name = name.split("Banana")[-1]
        name = name.split("ghildes")[-1]
        name = name.replace(".npy", "")
        name = name.replace("Deep_Sleep_SHHS", "SHHS_").replace("Deep_Sleep_snooze", "snooze_").replace("Deep_Sleep_philips", "philips_")
        name = name.replace("Deep_Sleep_combined_combined_dataset", "combined")
        name = name.replace("_to_", "_")

        df_name = tuple(name.split("_"))
        df.insert(0, df_name, data)

    return df


def name_replace_convnet(df, paths):
    for i, _ in enumerate(paths):
        data = np.load(paths[i])

        name = paths[i].name
        name = name.split("Banana")[-1]
        name = name.split("ghildes")[-1]
        name = name.replace(".npy", "")
        name = name.replace("Deep_Sleep_SHHS", "SHHS_").replace("Deep_Sleep_snooze", "snooze_")
        name = name.replace("Deep_Sleep_philips", "philips").replace("ConvNet_IID_combined", "combined")
        name = name.replace("_to_", "")

        df_name = tuple(name.split("_"))
        df.insert(0, df_name, data)

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
        df.insert(0, df_name, data)

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
        df.insert(0, df_name, data)

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
        df.insert(0, df_name, data)

    return df

if __name__ == '__main__':
    main()
