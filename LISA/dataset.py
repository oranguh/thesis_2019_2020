import torch
from torch.utils import data
import os
import numpy as np

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, folder):
        'Initialization'
        self.list_IDs = list_IDs
        self.folder = folder


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        folder_ = os.path.join(self.folder, ID)

        # 6609000

        X_ = torch.load(os.path.join(folder_, ID + '_data.pt'))
        Y_ = torch.load(os.path.join(folder_, ID + '_labels.pt'))

        X = np.zeros((X_.shape[0], 6806000))
        Y = np.full((Y_.shape[0], 6806000), -1)

        X[:X_.shape[0], :X_.shape[1]] = X_
        Y[:Y_.shape[0], :Y_.shape[1]] = Y_

        y_arousal = Y[0, :]
        # original: -1=unscored; 0=not_arousal; 1=arousal
        y_arousal += 1

        # sleep stage labels ['nonrem1', 'nonrem2', 'nonrem3', 'rem', 'undefined', 'wake']
        categories_ = [1, 2, 3, 4, 5, 6]
        y_sleep = np.multiply(Y[1:, :].transpose(), categories_).transpose().sum(axis=0)
        y_sleep[y_sleep < 0] = 0

        np.save("/project/marcoh/code/bsleep", y_sleep)
        np.save("/project/marcoh/code/brousal", y_arousal)

        return X, y_arousal, y_sleep
