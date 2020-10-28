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
import numpy as np

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

def calculatePadding(L, KSZ, S, D=1):
    '''
    :param L:       Input length (or width)
    :param KSZ:     Kernel size (or width)
    :param S:       Stride
    :param D:       Dilation Factor
    :return:        Returns padding such that output width is exactly half of input width
    '''

    # print(L, S, D, KSZ)
    pad = int(np.ceil(((L - 1) * S + D * (KSZ - 1) + 1 - L * 2)/2))
    # print("PAD", pad)
    # output_size = (L - 1) * S - 2 * pad + D * (KSZ - 1) + 1
    # print("OUTPUT SIZE", output_size)
    return pad


class PCC(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', pool=4, kernel=7):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activate = activation_func(activation)

        self.blocks = nn.Sequential(
            nn.MaxPool1d(pool),
            nn.Conv1d(in_channels, out_channels, kernel, padding=int(np.floor(kernel/2))),
            self.activate,
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel, padding=int(np.floor(kernel/2))),
            self.activate,
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):

        return self.blocks(x)


class UCC(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', kernel=7, pool=4, up_kernel=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.activate = activation_func(activation)

        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, up_kernel, stride=pool, padding=int(np.floor((up_kernel - pool)//2)))
        )

        self.blocks = nn.Sequential(
            # in_channels = concatanated size
            nn.Conv1d(out_channels*2, out_channels, kernel, padding=int(np.floor(kernel/2))),
            self.activate,
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(out_channels, out_channels, kernel, padding=int(np.floor(kernel/2))),
            self.activate,
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x, x2):

        # print("input", x.shape)
        x = self.up(x)
        # print(x.shape, x2.shape)
        x = torch.cat((x, x2), dim=1)
        # print(x.shape)
        x = self.blocks(x)
        # print(x.shape)

        return x


class Deep_Sleep(nn.Module):
    """

  """

    def __init__(self, channels_to_use):
        super(Deep_Sleep, self).__init__()
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem

        """
        self.INPUT_CHANNELS = channels_to_use
        self.AROUSAL_CLASSES = 3
        self.SLEEP_CLASSES = 6

        self.HIDDEN_SIZE = 15

        self.pcc1 = PCC(self.INPUT_CHANNELS, 15, pool=1)
        self.pcc2 = PCC(15, 18, pool=2)
        self.pcc3 = PCC(18, 21)
        self.pcc4 = PCC(21, 25)
        self.pcc5 = PCC(25, 30)
        self.pcc6 = PCC(30, 60)
        self.pcc7 = PCC(60, 120)
        self.pcc8 = PCC(120, 240)
        self.pcc9 = PCC(240, 480)

        self.ucc1 = UCC(480, 240)
        self.ucc2 = UCC(240, 120)
        self.ucc3 = UCC(120, 60)
        self.ucc4 = UCC(60, 30)
        self.ucc5 = UCC(30, 25)
        self.ucc6 = UCC(25, 21)
        self.ucc7 = UCC(21, 18)
        self.ucc8 = UCC(18, self.HIDDEN_SIZE, pool=2, up_kernel=4)

        self.arousal_classifier = nn.Sequential(
            nn.Conv1d(self.HIDDEN_SIZE, self.AROUSAL_CLASSES, 1)
        )

        self.sleep_classifier = nn.Sequential(
            nn.Conv1d(self.HIDDEN_SIZE, self.SLEEP_CLASSES, 1)
        )

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
        # print("input", x.shape)
        pcc1 = self.pcc1(x)
        # print("pcc1 ", pcc1.shape)
        pcc2 = self.pcc2.forward(pcc1)
        # print("pcc2 ", pcc2.shape)
        pcc3 = self.pcc3.forward(pcc2)
        # print("pcc3 ", pcc3.shape)
        pcc4 = self.pcc4.forward(pcc3)
        # print("pcc4 ", pcc4.shape)
        pcc5 = self.pcc5.forward(pcc4)
        # print("pcc4 ", pcc5.shape)
        pcc6 = self.pcc6.forward(pcc5)
        # print("pcc4 ", pcc6.shape)
        pcc7 = self.pcc7.forward(pcc6)
        # print("pcc4 ", pcc7.shape)
        pcc8 = self.pcc8.forward(pcc7)
        # print("pcc4 ", pcc8.shape)
        pcc9 = self.pcc9.forward(pcc8)
        # print("pcc9 ", pcc9.shape)

        ucc1 = self.ucc1.forward(pcc9, pcc8)
        # print("ucc1 ", ucc1.shape)
        ucc2 = self.ucc2.forward(ucc1, pcc7)
        # print("ucc2 ", ucc2.shape)
        ucc3 = self.ucc3.forward(ucc2, pcc6)
        # print("ucc3 ", ucc3.shape)
        ucc4 = self.ucc4.forward(ucc3, pcc5)
        # print("ucc4 ", ucc4.shape)
        ucc5 = self.ucc5.forward(ucc4, pcc4)
        # print("ucc5 ", ucc5.shape)
        ucc6 = self.ucc6.forward(ucc5, pcc3)
        # print("ucc6 ", ucc6.shape)
        ucc7 = self.ucc7.forward(ucc6, pcc2)
        # print("ucc7 ", ucc7.shape)
        ucc8 = self.ucc8.forward(ucc7, pcc1)
        # print("ucc8 ", ucc8.shape)
        out_arousal = self.arousal_classifier(ucc8)
        out_sleep = self.sleep_classifier(ucc8)

        # print("DONE", out_arousal.shape, out_sleep.shape)
        return out_arousal, out_sleep
