import sys
import os
import torch
import torchvision
import numpy as np
from torchvision.datasets import CelebA
from torch.utils.data import Subset

white = np.load(os.path.expanduser('~/post_hoc_debiasing/celebrace/white_full.npy'))
black = np.load(os.path.expanduser('~/post_hoc_debiasing/celebrace/black_full.npy'))
asian = np.load(os.path.expanduser('~/post_hoc_debiasing/celebrace/asian_full.npy'))


class CelebRace(CelebA):

    def __getitem__(self, index):

        X, target = super().__getitem__(index)

        augment = torch.tensor([white[index] > .501, black[index] > .501, asian[index] > .501, index+1], dtype=torch.long)

        return X, torch.cat((target, augment))


def unambiguous(dataset, split='train', thresh=.7):

    if split == 'train':
        n = 162770
    else:
        n = 19962
    unambiguous_indices = [i for i in range(n) if (white[i] > thresh or black[i] > thresh or asian[i] > thresh)]

    return Subset(dataset, unambiguous_indices)


def split_check(dataset, split='train', thresh=.7):

    if split == 'train':
        n = 162770
    else:
        n = 19962
    unambiguous_indices = [i for i in range(n) if (asian[i] > thresh)]

    return Subset(dataset, unambiguous_indices)

