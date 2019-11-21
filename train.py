import torch
import numpy as np
from torch.utils import data
from tools import accuracy, CustomFormatter
from dataset import Dataset
from visualizations import plot_confusion_matrix, plot_classes_distribution
from model import ConvNet
import torch.nn as nn
import torch.optim as optim
import os
import pickle as pkl
from sklearn import metrics
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def load_obj(name ):
    with open(name, 'rb') as f:
        return pkl.load(f)


# create logger with 'spam_application'
logger = logging.getLogger("My_app")
logger.setLevel(logging.INFO)

writer = SummaryWriter(comment=str(datetime.now().strftime("%Y%m%d-%H%M%S")))

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)


# load metadata stuff
data_folder = 'data_2/'
sleep_stages = ['N1', 'N2', 'N3/4', 'REM']

partition = load_obj(os.path.join(data_folder, 'partition.pkl'))
labels = load_obj(os.path.join(data_folder, 'labels.pkl'))

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# device = 'cpu'
# cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 0}
max_epochs = 10

# Generators

training_set = Dataset(partition['train'], labels, data_folder)
training_generator = data.DataLoader(training_set, **params)

validation_set = Dataset(partition['validation'], labels, data_folder)
validation_generator = data.DataLoader(validation_set, **params)

# models

network = ConvNet(1, 4)
network.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=1e-4)

# Loop over epochs
for epoch in range(max_epochs):
    # Training
    true_array = np.empty(0)
    pred_array = np.empty(0)
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        # print(local_batch.shape, local_labels.shape)

        batch_sz_ = local_batch.shape[0]
        data_sz = local_batch.shape[1]
        x = local_batch.view([batch_sz_, 1, data_sz]).type(torch.cuda.FloatTensor).to(device)
        y = local_labels.type(torch.cuda.LongTensor).to(device)

        out = network.forward(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        true_array = np.asarray([*true_array, *y.cpu().numpy()])
        pred_array = np.asarray([*pred_array, *out.argmax(dim=1).cpu().numpy()])
    acc = accuracy(pred_array, true_array)
    writer.add_scalar('Loss/train', loss.item(), global_step=epoch)
    writer.add_scalar('Accuracy/train', acc, global_step=epoch)
    report = metrics.classification_report(true_array, pred_array, target_names=sleep_stages)
    writer.add_text('Report Training', report + '\n', global_step=epoch)
    logger.info("TRAINING: epoch: {} Loss {:.3f} Accuracy: {:.3f}".format(epoch, loss.item(), acc))
    logger.info("TRAINING Classification Report: \n{}\n".format(report))

    # Validation
    with torch.set_grad_enabled(False):
        true_array = np.empty(0)
        pred_array = np.empty(0)
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            batch_sz_ = local_batch.shape[0]
            data_sz = local_batch.shape[1]
            x = local_batch.view([batch_sz_, 1, data_sz]).type(torch.cuda.FloatTensor).to(device)
            y = local_labels.type(torch.cuda.LongTensor).to(device)

            out = network.forward(x)
            loss = criterion(out, y)

            true_array = np.asarray([*true_array, *y.cpu().numpy()])
            pred_array = np.asarray([*pred_array, *out.argmax(dim=1).cpu().numpy()])

        acc = accuracy(pred_array, true_array)
        # print(true_array, pred_array)
        report = metrics.classification_report(true_array, pred_array, target_names=sleep_stages)
        writer.add_text('Report Validation', report + '\n', global_step=epoch)
        writer.add_scalar('Loss/validation', loss.item(), global_step=epoch)
        writer.add_scalar('Accuracy/validation', acc, global_step=epoch)
        logger.info("VALIDATION: epoch: {} Loss {:.3f} Accuracy: {:.3f}".format(epoch, loss.item(), acc))
        logger.info("VALIDATION Classification Report: \n{}\n".format(report))



np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# print(y_test, "\n", y_pred)

# Plot normalized confusion matrix
fig_confusion = plot_confusion_matrix(true_array, pred_array, classes=sleep_stages, normalize=True, title='Normalized confusion matrix')
writer.add_figure("Confusion Matrix", fig_confusion)

fig_classes = plot_classes_distribution(labels, classes=sleep_stages)
writer.add_figure("Class distribution", fig_classes)

writer.close()