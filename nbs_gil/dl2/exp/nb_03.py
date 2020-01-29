
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/03_minibatch_training.ipynb

from exp.nb_02 import *
import torch.nn.functional as F

def accuracy(out, yb):
    return (torch.argmax(out, dim=-1) == yb).float().mean()

from torch import optim

class Dataset():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
           DataLoader(valid_ds, batch_size=bs*2, shuffle=False, **kwargs))