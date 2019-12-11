import torch
from torch.utils import data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, folder):
        'Initialization'
        self.list_IDs_labels = labels
        self.list_IDs = list_IDs
        self.folder = folder


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        label_ID = self.list_IDs_labels[index]

        # Load data and get label
        # print(ID)
        # print(asdasd)
        # print('data/' + str(ID) + '.pt')
        X = torch.load(self.folder + ID + '.pt')
        # print(X)
        # print(asdasd)
        Y = torch.load(self.folder + label_ID + '.pt')

        return X, Y
