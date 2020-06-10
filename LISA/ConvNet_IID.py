"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict


class ConvNet_IID(nn.Module):
    """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

    def __init__(self, channels_to_use):
        super(ConvNet_IID, self).__init__()
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem

        """
        self.AROUSAL_CLASSES = 3
        self.SLEEP_CLASSES = 6

        self.n_classes = channels_to_use
        self.layers = OrderedDict()

        self.layers["conv_1"] = nn.Conv1d(channels_to_use, 64, 8, stride=1, padding=1)
        self.layers["batchnorm_1"] = nn.BatchNorm1d(64)
        self.layers["relu_1"] = nn.ReLU(inplace=True)
        self.layers["maxpool_1"] = nn.MaxPool1d(8, stride=4, padding=1)

        self.layers["conv_2"] = nn.Conv1d(64, 128, 8, stride=1, padding=1)
        self.layers["batchnorm_2"] = nn.BatchNorm1d(128)
        self.layers["relu_2"] = nn.ReLU(inplace=True)
        self.layers["maxpool_2"] = nn.MaxPool1d(8, stride=4, padding=1)

        self.layers["conv_3"] = nn.Conv1d(128, 256, 8, stride=1, padding=1)
        self.layers["batchnorm_3"] = nn.BatchNorm1d(256)
        self.layers["relu_3"] = nn.ReLU(inplace=True)
        self.layers["maxpool_3"] = nn.MaxPool1d(8, stride=4, padding=1)

        self.layers["conv_4"] = nn.Conv1d(256, 512, 8, stride=1, padding=1)
        self.layers["batchnorm_4"] = nn.BatchNorm1d(512)
        self.layers["relu_4"] = nn.ReLU(inplace=True)
        self.layers["maxpool_4"] = nn.MaxPool1d(8, stride=2, padding=1)

        self.layers["conv_5"] = nn.Conv1d(512, 1024, 8, stride=1, padding=1)
        self.layers["batchnorm_5"] = nn.BatchNorm1d(1024)
        self.layers["relu_5"] = nn.ReLU(inplace=True)
        self.layers["maxpool_5"] = nn.MaxPool1d(8, stride=2, padding=1)

        # self.layers["conv_6"] = nn.Conv1d(1024, 2048, 8, stride=1, padding=1)
        # self.layers["batchnorm_6"] = nn.BatchNorm1d(2048)
        # self.layers["relu_6"] = nn.ReLU(inplace=True)
        # self.layers["maxpool_6"] = nn.MaxPool1d(8, stride=2, padding=1)

        self.convoluter = nn.Sequential(self.layers)
        # for 30 seconds 200Hz this is 16384
        # 30 seconds 100 Hz is 4096?
        unknown = 4096
        self.classifier = nn.Sequential(
            nn.Linear(unknown, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, self.AROUSAL_CLASSES))

        self.classifier_aux = nn.Sequential(
            nn.Linear(unknown, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, self.SLEEP_CLASSES))

        print("Created model : {}".format(self))

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        """

        out = self.convoluter(x)

        # desired = (BATCHSIZE, 512)
        # print(out.shape)
        out = out.view(x.shape[0], -1)
        # print(out.shape)

        out1 = self.classifier(out)
        out2 = self.classifier_aux(out)

        return out1, out2
