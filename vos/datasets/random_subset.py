import torch
from torch.utils.data import Dataset

from numpy.random import randint

from vos.utils.quick_args import save__init__args

class RandomSubset(Dataset):
    """ Due the limitation of loading and testing the val dataset.
    A wrapper that surve as a subset for the every dataset. The provided dataset should be
    indexable.
    """
    def __init__(self, dataset,
        subset_len= 10,
        ):
        save__init__args(locals(), underscore= True)
        # reset and recording
        self._n_getitem_ = 0

        self.sample_subset()

    def sample_subset(self):
        """ randomly sample a subset of the given dataset
        """
        data_len = len(self._dataset)
        self._idxs = randint(data_len, size= self._subset_len)

    def __len__(self):
        return self._subset_len

    def __getitem__(self, idx):
        self._n_getitem_ += 1
        item = self._dataset[self._idxs[idx]]
        
        if self._n_getitem_ % self._subset_len == 0:
            self.sample_subset()
        
        return item
