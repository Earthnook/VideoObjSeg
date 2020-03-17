import numpy as np
import torch
from torch.utils.data import Dataset

class MultiDataset(Dataset):
    """ A dataset that serve as multi-dataset, that merge all given datasets as one.
    Think all data are put in a array connectively.
    """
    def __init__(self, *datasets):
        self._datasets = datasets
        super().__init__()

    def __len__(self):
        return sum([len(d) for d in self._datasets])

    def __getitem__(self, idx):
        for d in self._datasets:
            length = len(d)
            if idx < length:
                return d[idx]
            else:
                idx -= length
        raise ValueError("Index is larger than dataset length")
