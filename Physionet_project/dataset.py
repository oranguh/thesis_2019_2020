import torch
from torch.utils import data

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

        X = torch.load(self.folder + ID + '_data.pt')

        Y = torch.load(self.folder + ID + '_labels.pt')

        return X, Y
