import sys
import os
import torch
import torchvision
import numpy as np
from torchvision.datasets import CelebA
from torch.utils.data import Subset

thresh = .7
white = np.load(os.path.expanduser('~/post_hoc_debiasing/white_full.npy'))
black = np.load(os.path.expanduser('~/post_hoc_debiasing/black_full.npy'))       
asian = np.load(os.path.expanduser('~/post_hoc_debiasing/asian_full.npy'))


class CelebRace(CelebA):

#    def __init__(self, transform, split, root='./data', download=True):
#        super().__init__(transform=transform, split=split, root=root, download=download)
        
    def __getitem__(self, index):

        X, target = super().__getitem__(index)

        race = [white[index] > .501, black[index] > .501, asian[index] > .501]
        new = torch.tensor(race, dtype=torch.long)

        return X, torch.cat((target, new))


def unambiguous_bw(dataset, split='train'):

    if split == 'train':
        n = 162770
    else:
        n = 19962
    unambiguous_indices = [i for i in range(n) if (white[i] > thresh or black[i] > thresh)]

    return Subset(dataset, unambiguous_indices)


def split_check(dataset, split='train'):

    if split == 'train':
        n = 162770
    else:
        n = 19962

    unambiguous_indices = [i for i in range(n) if (asian[i] > thresh)]

    return Subset(dataset, unambiguous_indices)

