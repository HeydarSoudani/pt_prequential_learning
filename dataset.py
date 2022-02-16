import torch
from torch import tensor
from torch.utils.data import Dataset

import numpy as np

DATASETS = {'pmnist', 'rmnist', 'mnist', 'sea'}

def onehot2index(data):
  # input >> data: [nClass, ]
  return np.argmax(data)

class ChunkDataset(Dataset):
  def __init__(self, data, args, transforms=None):
  
    # if args.dataset in ['mnist', 'pmnist', 'rmnist', 'fmnist', 'pfmnist', 'rfmnist']:
    #   self.tensor_view = (1, 28, 28)
    # elif args.dataset in ['cifar10', 'cifar100']:
    #   self.tensor_view = (3, 32, 32)
    self.transforms = transforms
    self.data = []
    self.labels = np.argmax(data[:, -10:], axis=1)
    self.label_set = set(self.labels)
  
    for idx, s in enumerate(data):
      # x = (tensor(s[:-1], dtype=torch.float) / 255).view(self.tensor_view)
      # y = tensor(self.labels[idx], dtype=torch.long)
      x = (tensor(s[:-10], dtype=torch.float) / 255)
      y = tensor(onehot2index(s[-10:]), dtype=torch.long)
      self.data.append((x, y))

  def __getitem__(self, index):
    if self.transforms != None:
      sample, label = self.data[index]
      sample = self.transforms(sample)
      return sample, label
    else:
      return self.data[index]

  def __len__(self):
    return len(self.data)
  