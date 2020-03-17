""" Pytorch interface for ECSSD dataset
The expected file structure is:
    root/:
        - images/:
            ... image files
        - ground_truth_mask/:
            ... image files (with same file name as images)
"""
import os
from os import path

import numpy as np
import torch
from torch.utils.data import Dataset

from skimage import io as skio

IMAGE_NAME = "images"
MASK_NAME = "ground_truth_mask"

class ECSSD(Dataset):
    def __init__(self, root, portion= (0., 1.0)):
        self._root = root
        self._portion = portion

        image_filenames = os.listdir(path.join(self._root, IMAGE_NAME))
        mask_filenames = os.listdir(path.join(self._root, MASK_NAME))
        assert len(image_filenames) == len(mask_filenames), "Inconsistent data length"

        self.filenames = image_filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = skio.imread(path.join(self._root, IMAGE_NAME, self.filenames[idx])) / 255
        mask = skio.imread(path.join(self._root, MASK_NAME, self.filenames[idx]))

        image = image.astype(np.float32)
        image = image.transpose(2,0,1)
        mask = mask.astype(np.uint8)
        mask = np.stack([1-mask, mask])

        return dict(
            image = torch.from_numpy(image),
            mask = torch.from_numpy(mask),
            n_objects = torch.Tensor([1,])[0], # if just a number, it will generate a random vector
        )
        
        