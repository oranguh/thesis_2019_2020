"""
Trying to replicate Howe-Patterson model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict


def activation_func(activation):
    return nn.ModuleDict({
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01, inplace=True),
        'selu': nn.SELU(inplace=True),
        'tanh': nn.Tanh(),
        'none': nn.Identity()
    })[activation]


class DCU1(nn.Module):
    def __init__(self, in_channels, out_channels, activation='selu', dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation

        self.activate = activation_func(activation)

        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 51, stride=1, padding=1, groups=in_channels, dilation=dilation),
            nn.Conv1d(in_channels, out_channels * 4, 1, stride=1, padding=1, groups=1, dilation=dilation),
            self.activate,
            nn.BatchNorm1d(out_channels * 4),
            nn.Conv1d(out_channels * 4, out_channels * 4, 1, stride=1, padding=1, groups=out_channels * 4,
                      dilation=dilation),
            nn.Conv1d(out_channels * 4, out_channels, 1, stride=1, padding=1, groups=1, dilation=dilation),
            self.activate,
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        input = x

        x = self.blocks(x)
        x = torch.cat((x, input))
        return x


class DCU2(nn.Module):
    def __init__(self, in_channels, out_channels, activation='selu', dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation

        self.activate = activation_func(activation)

        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 25, stride=1, padding=1, groups=in_channels, dilation=dilation),
            nn.Conv1d(in_channels, out_channels * 4, 1, stride=1, padding=1, groups=1, dilation=dilation),
            self.activate,
            nn.BatchNorm1d(out_channels * 4),
            nn.Conv1d(out_channels * 4, out_channels * 4, 1, stride=1, padding=1, groups=out_channels * 4,
                      dilation=dilation),
            nn.Conv1d(out_channels * 4, out_channels, 1, stride=1, padding=1, groups=1, dilation=dilation),
            self.activate,
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        input = x

        x = self.blocks(x)
        x = torch.cat((x, input))
        return x


class Howe_Patterson(nn.Module):
    """

  """

    def __init__(self, n_channels, n_classes):
        super(Howe_Patterson, self).__init__()
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem

        """
        self.n_classes = n_classes
        self.layers = OrderedDict()

        self.layers["conv_1"] = DCU1(12, 24)
        self.layers["maxpool_1"] = nn.MaxPool1d(2, stride=1, padding=1)
        self.layers["conv_2"] = DCU1(36, 24)
        self.layers["maxpool_2"] = nn.MaxPool1d(5, stride=1, padding=1)
        self.layers["conv_3"] = DCU1(60, 24)
        self.layers["maxpool_3"] = nn.MaxPool1d(5, stride=1, padding=1)

        self.layers["conv_4"] = DCU2(84, 24, dilation=1)
        self.layers["conv_5"] = DCU2(108, 24, dilation=2)
        self.layers["conv_6"] = DCU2(132, 24, dilation=4)
        self.layers["conv_7"] = DCU2(156, 24, dilation=8)
        self.layers["conv_8"] = DCU2(180, 24, dilation=16)
        self.layers["conv_9"] = DCU2(204, 24, dilation=32)
        self.layers["conv_10"] = DCU2(228, 24, dilation=16)
        self.layers["conv_11"] = DCU2(252, 24, dilation=8)
        self.layers["conv_12"] = DCU2(276, 24, dilation=4)
        self.layers["conv_13"] = DCU2(300, 24, dilation=2)
        self.layers["conv_14"] = DCU2(324, 24, dilation=1)

        self.convoluter = nn.Sequential(self.layers)

        self.lstm_conv1 = nn.Conv1d(348, 128, 1)
        self.lstm_conv2 = nn.Conv1d(256, 128, 1)
        self.lstm_conv3 = nn.Conv1d(128, 4, 1)

        self.lstm = nn.LSTM(348, 128, 1, bidirectional=True)

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

        # hidden = torch.zeros(128)
        out = self.convoluter(x)

        out1 = self.lstm_conv1(out)

        out2, hidden = self.lstm(out)
        out2 = self.lstm_conv2(out2)

        # not sure if addition or concatanation?
        out = out1 + out2
        out = out * (1 / (2 ** 0.5))
        out = F.tanh(out)

        out = self.lstm_conv3(out)

        return out
