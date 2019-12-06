import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os

from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print("{}".format(cm))

    fig, ax = plt.subplots()
    # cmap = sns.diverging_palette(220, 20, n=7)
    cmap = 'RdYlGn'
    sns.heatmap(cm, annot=True, ax=ax, center=1/len(classes), cmap=cmap, robust=True, cbar=True)  # annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    ax.set_ylim(len(classes) + 0.5, -0.5)


    # plt.tight_layout()
    plt.savefig(os.path.join('figures/', 'confusion_matrix.png'))
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
