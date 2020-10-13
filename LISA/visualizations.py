import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os
from pathlib import Path


from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          title=None,
                          labels=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    sub_titles = ['Normalized confusion matrix', 'Confusion matrix, without normalization']
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('{}'.format(title), fontsize=16)
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    cmap = 'RdYlGn'

    for i, normalization_flag in enumerate([True, False]):

        if normalization_flag:
            cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.1%'
        else:
            cm_plot = cm
            fmt = 'd'

        # cmap = sns.diverging_palette(220, 20, n=7)

        # sns.heatmap(cm, annot=True, ax=ax, center=1/len(classes), cmap=cmap, robust=True, cbar=True)  # annot=True to annotate cells
        sns.heatmap(cm_plot, annot=True, ax=ax[i], cmap=cmap, robust=True, cbar=True, fmt=fmt)
        # labels, title and ticks
        ax[i].set_xlabel('Predicted labels')
        ax[i].set_ylabel('True labels')
        ax[i].set_title(sub_titles[i])
        ax[i].xaxis.set_ticklabels(classes)
        ax[i].yaxis.set_ticklabels(classes)
        # ax.set_ylim(len(classes) + 0.5, -0.5)


    # plt.tight_layout()

    Save_folder = Path("figures/confusions") / "_".join(title.split("_")[:-1])
    Save_folder.mkdir(parents=True, exist_ok=True)

    filename = title + '.png'
    plt.savefig(Save_folder / filename)
    # plt.show()

    return fig


def sleep_staging_confusion(cm_dict,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    sub_titles = ['Normalized confusion matrix', 'Confusion matrix']
    fig, ax = plt.subplots(7, 2, figsize=(20, 10), sharex="all", sharey="all")
    fig.suptitle('{}'.format(title), fontsize=16)
    # Compute confusion matrix
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    cmap = 'RdYlGn'
    cm_total = np.asarray(list(cm_dict.values())).sum(axis=0)
    for j, stage in enumerate(["N1", "N2", "N3", "REM", "Undefined", "Wake", "Sum"]):
        if stage == "Sum":
            cm = cm_total
        # elif stage == "Undefined":
        #     j -= 1
        #     continue
        else:
            cm = cm_dict[stage]
        for i, normalization_flag in enumerate([True, False]):

            if normalization_flag:
                cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.1%'
            else:
                cm_plot = cm
                fmt = 'd'

            # cmap = sns.diverging_palette(220, 20, n=7)

            # sns.heatmap(cm, annot=True, ax=ax, center=1/len(classes), cmap=cmap, robust=True, cbar=True)  # annot=True to annotate cells
            sns.heatmap(cm_plot, annot=True, ax=ax[j, i], cmap=cmap, robust=True, cbar=False, fmt=fmt)
            # labels, title and ticks
            # ax[j, i].set_xlabel('Predicted labels')
            # ax[j, i].set_ylabel('True labels')
            # print(cm.sum(axis=1)/cm_total.sum(axis=1))
            blup = "{stage}: {sleep_total:.0f}% of sleep; arousal: {conf[1]:.0f}%".format(stage=stage,
                                                                                   sleep_total= cm.sum()/cm_total.sum()*100,
                                                                                   conf=cm.sum(axis=1)/cm_total.sum(axis=1)*100)
            ax[j, i].set_title(blup)
            ax[j, i].xaxis.set_ticklabels(["Pred not arousal", "Pred arousal"])
            ax[j, i].yaxis.set_ticklabels(["True not arousal", "True arousal"], rotation=45, rotation_mode="anchor")

            # ax.set_ylim(len(classes) + 0.5, -0.5)


    # plt.tight_layout()

    Save_folder = Path("figures/confusions") / title
    Save_folder.mkdir(parents=True, exist_ok=True)

    filename = title + '.png'
    plt.savefig(Save_folder / filename)
    # plt.show()

    return fig


def plot_classes_distribution(labels, classes):

    # plt.bar(list(labels.keys()), labels.values(), 1, color='g')
    # ax = plt.gca()
    # ax.xaxis.set_ticklabels(['N1', 'N2', 'N3/4', 'REM'])
    # plt.show()

    fig, ax = plt.subplots()

    df = pd.DataFrame(labels.values(), labels.keys())
    df = df.rename(columns={0: 'Classes'})
    sns.countplot(x='Classes', data=df)
    # ax = plt.gca()
    ax.xaxis.set_ticklabels(classes)
    plt.savefig(os.path.join('figures/', 'class_distribution.png'))
    plt.show()

    return fig
