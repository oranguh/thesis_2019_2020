import torch
import os
import pickle as pkl

def load_obj(name):
    with open(name, 'rb') as f:
        return pkl.load(f)


data_folder = 'E:\\data\\you-snooze-you-win-the-physionet-computing-in-cardiology-challenge-2018-1.0.0\\marco'
partition = load_obj('E:\\data\\you-snooze-you-win-the-physionet-computing-in-cardiology-challenge-2018-1.0.0\\marco\\data_partition.pkl')

vali = partition["validation"]
traini = partition["train"]
print(len(vali))
print(len(traini))

max_size = 0

for item in vali:
    print(item)
    folder = os.path.join(data_folder, item)
    X_ = torch.load(os.path.join(folder, item + '_data.pt'))
    Y_ = torch.load(os.path.join(folder, item + '_labels.pt'))
    print(X_.shape, Y_.shape)
    if max_size < X_.shape[-1]:
        max_size = X_.shape[-1]
        print(max_size)

for item in traini:
    print(item)
    folder = os.path.join(data_folder, item)
    X_ = torch.load(os.path.join(folder, item + '_data.pt'))
    Y_ = torch.load(os.path.join(folder, item + '_labels.pt'))
    print(X_.shape, Y_.shape)
    if max_size < X_.shape[-1]:
        max_size = X_.shape[-1]
        print(max_size)


