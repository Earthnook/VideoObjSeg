""" Interface of MSRA10K Salient Object Database
https://mmcheng.net/msra10k/
The file structure should be 
    root/:
        - Imgs/:
            ... image files and mask files
        - Readme.txt
"""
import os
from os import path

import numpy as np
import torch
from torch.utils.data import Dataset

from skimage import io as skio

SUB_DIR = "Imgs"

class MSRA10K(Dataset):
    def __init__(self, root, portion= (0., 1.0)):
        self._root = root
        self._portion = portion

        img_files = os.listdir(path.join(self._root, SUB_DIR))
        self.img_files = sorted([i[:-4] for i in img_files if ".png" in i])

    def __len__(self):
        return int(len(self.img_files) * (self._portion[1] - self._portion[0]))

    def __getitem__(self, idx):
        idx += int(len(self.img_files) * self._portion[0])

        image = skio.imread(path.join(self._root, SUB_DIR, self.img_files[idx]) + ".jpg") / 255
        mask = skio.imread(path.join(self._root, SUB_DIR, self.img_files[idx]) + ".png") / 255

        image = image.astype(np.float32)
        # incase of gray-scale image
        if len(image.shape) == 2:
            image = np.tile(image, (3,1,1))
        elif len(image.shape) == 3:
            image = image.transpose(2,0,1)
        else:
            raise ValueError("Wrong image shape dimensions\n{}".format(str(self.img_files[idx])))
        mask = mask.astype(np.uint8)
        mask = np.stack([1-mask, mask])

        return dict(
            image = torch.from_numpy(image),
            mask = torch.from_numpy(mask),
            n_objects = torch.LongTensor([int(1)])[0]
        )
