import sys
import os
import torch
import torchvision
import numpy as np
from torchvision.datasets import CelebA


class CelebRace(CelebA):

    def __init__(self, transform, split, root='./data', download=True):

        self.white = np.load(os.path.expanduser('~/post_hoc_debiasing/white_100k.npy'))
        self.black = np.load(os.path.expanduser('~/post_hoc_debiasing/black_100k.npy'))       
        self.asian = np.load(os.path.expanduser('~/post_hoc_debiasing/asian_100k.npy'))

        super().__init__(transform=transform, split=split, root=root, download=download)
        
    def __getitem__(self, index):

        X, target = super().__getitem__(index)

        if index < 100000:
            race = [self.white[index] > .6, self.black[index] > .6, self.asian[index] > .6]
            new = torch.tensor(race, dtype=torch.long)
        else:
            new = torch.tensor([1, 0, 0], dtype=torch.long)

        return X, torch.cat((target, new))

