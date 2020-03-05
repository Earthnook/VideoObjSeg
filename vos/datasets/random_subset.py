import torch
from torch.utils.data import Dataset

from numpy.random import choice

from vos.utils.quick_args import save__init__args
from vos.utils.image_shaper import random_crop

class RandomSubset(Dataset):
    """ Due the limitation of loading and testing the val dataset.
    A wrapper that surve as a subset for the every dataset. The provided dataset should be
    indexable.
    """
    def __init__(self, dataset,
            subset_len= 10,
            resolution= (384, 384), # control the output of the image, to make a batch
            resize_method= "crop", # choose between "crop", "resize"
        ):
        save__init__args(locals(), underscore= True)
        # reset and recording
        self._n_getitem_ = 0

        self.sample_subset()

    def sample_subset(self):
        """ randomly sample a subset of the given dataset
        """
        data_len = self._dataset.__len__()
        self._idxs = choice(range(data_len), size= self._subset_len, replace= False)

    def __len__(self):
        return self._subset_len

    def __getitem__(self, idx):
        self._n_getitem_ += 1
        item = self._dataset.__getitem__(self._idxs[idx])
        
        if self._n_getitem_ % self._subset_len == 0:
            self.sample_subset()

        if self._resize_method == "crop":
            item["video"], item["mask"] = random_crop(self._resolution, item["video"], item["mask"])
        elif self._resize_method == "resize":
            raise NotImplementedError # put here for later implementation
        else:
            pass
        
        return item
