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


def primary_school_maths(dilation, kernel):

    padding = 0.5*dilation*(kernel - 1)

    return int(padding)

class DCU1(nn.Module):
    def __init__(self, in_channels, out_channels, activation='selu', dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation

        self.activate = activation_func(activation)


        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 51, padding=primary_school_maths(dilation, 51), stride=1, groups=in_channels, dilation=dilation),
            nn.Conv1d(in_channels, out_channels * 4, 1, stride=1, groups=1, dilation=dilation),
            self.activate,
            nn.BatchNorm1d(out_channels * 4),
            nn.Conv1d(out_channels * 4, out_channels * 4, 1, stride=1, groups=out_channels * 4,
                      dilation=dilation),
            nn.Conv1d(out_channels * 4, out_channels, 1, stride=1, groups=1, dilation=dilation),
            self.activate,
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        input = x
        # print("DCU1", x.shape)
        x = self.blocks(x)
        x = torch.cat((x, input), dim=1)
        return x


class DCU2(nn.Module):
    def __init__(self, in_channels, out_channels, activation='selu', dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation

        self.activate = activation_func(activation)

        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 25, padding=primary_school_maths(dilation, 25), stride=1, groups=in_channels, dilation=dilation),
            nn.Conv1d(in_channels, out_channels * 4, 1, padding=primary_school_maths(dilation, 1), stride=1, groups=1, dilation=dilation),
            self.activate,
            nn.BatchNorm1d(out_channels * 4),
            nn.Conv1d(out_channels * 4, out_channels * 4, 1, padding=primary_school_maths(dilation, 1), stride=1, groups=out_channels * 4,
                      dilation=dilation),
            nn.Conv1d(out_channels * 4, out_channels, 1, padding=primary_school_maths(dilation, 1), stride=1, groups=1, dilation=dilation),
            self.activate,
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        input = x
        # print("DCU2", x.shape)
        x = self.blocks(x)
        x = torch.cat((x, input), dim=1)

        return x


class Howe_Patterson(nn.Module):
    """

  """

    def __init__(self, channels_to_use):
        super(Howe_Patterson, self).__init__()
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem

        """
        self.INPUT_CHANNELS = channels_to_use
        self.AROUSAL_CLASSES = 3
        self.SLEEP_CLASSES = 6

        self.HIDDEN_SIZE = 128

        self.layers = OrderedDict()

        self.layers["conv_1"] = DCU1(self.INPUT_CHANNELS, self.INPUT_CHANNELS*2)
        self.layers["maxpool_1"] = nn.MaxPool1d(2, stride=2, padding=0, ceil_mode=True)
        self.layers["conv_2"] = DCU1(self.INPUT_CHANNELS*3, self.INPUT_CHANNELS*2)
        self.layers["maxpool_2"] = nn.MaxPool1d(5, stride=5, padding=0, ceil_mode=True)
        self.layers["conv_3"] = DCU1(self.INPUT_CHANNELS*5, self.INPUT_CHANNELS*2)
        self.layers["maxpool_3"] = nn.MaxPool1d(5, stride=5, padding=0, ceil_mode=True)

        self.layers["conv_4"] = DCU2(self.INPUT_CHANNELS*7, self.INPUT_CHANNELS*2, dilation=1)
        self.layers["conv_5"] = DCU2(self.INPUT_CHANNELS*9, self.INPUT_CHANNELS*2, dilation=2)
        self.layers["conv_6"] = DCU2(self.INPUT_CHANNELS*11, self.INPUT_CHANNELS*2, dilation=4)
        self.layers["conv_7"] = DCU2(self.INPUT_CHANNELS*13, self.INPUT_CHANNELS*2, dilation=8)
        self.layers["conv_8"] = DCU2(self.INPUT_CHANNELS*15, self.INPUT_CHANNELS*2, dilation=16)
        self.layers["conv_9"] = DCU2(self.INPUT_CHANNELS*17, self.INPUT_CHANNELS*2, dilation=32)
        self.layers["conv_10"] = DCU2(self.INPUT_CHANNELS*19, self.INPUT_CHANNELS*2, dilation=16)
        self.layers["conv_11"] = DCU2(self.INPUT_CHANNELS*21, self.INPUT_CHANNELS*2, dilation=8)
        self.layers["conv_12"] = DCU2(self.INPUT_CHANNELS*23, self.INPUT_CHANNELS*2, dilation=4)
        self.layers["conv_13"] = DCU2(self.INPUT_CHANNELS*25, self.INPUT_CHANNELS*2, dilation=2)
        self.layers["conv_14"] = DCU2(self.INPUT_CHANNELS*27, self.INPUT_CHANNELS*2, dilation=1)

        self.convoluter = nn.Sequential(self.layers)

        self.lstm_conv1 = nn.Conv1d(self.INPUT_CHANNELS*29, self.HIDDEN_SIZE, 1)
        self.lstm = nn.LSTM(self.INPUT_CHANNELS*29, self.HIDDEN_SIZE, 1, bidirectional=True)

        self.lstm_conv2 = nn.Conv1d(self.HIDDEN_SIZE*2, self.HIDDEN_SIZE, 1)

        # The second argument denotes the output classes
        self.lstm_conv3 = nn.Conv1d(self.HIDDEN_SIZE, self.AROUSAL_CLASSES, 1)

        self.lstm_conv3_2 = nn.Conv1d(self.HIDDEN_SIZE, self.SLEEP_CLASSES, 1)



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
        # print(out.shape)
        batch = out.shape[0]
        sequence_length = out.shape[2]
        channel_size = out.shape[1]
        out1 = self.lstm_conv1(out)
        # print(out1.shape)
        out2, hidden = self.lstm(out.view(sequence_length, batch, channel_size))
        out2 = self.lstm_conv2(out2.view(batch, self.HIDDEN_SIZE*2, sequence_length))

        # not sure if addition or concatanation?
        out = out1 + out2
        out = out * (1 / (2 ** 0.5))
        out = torch.tanh(out)

        # output for arousals
        out_arousal = self.lstm_conv3(out)

        # output for sleep staging
        out_sleep = self.lstm_conv3_2(out)

        # print("DONE", out.shape)
        return out_arousal, out_sleep
