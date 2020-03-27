"""
dataset from http://home.bharathh.info/pubs/codes/SBD/download.html
The directory structure should be like (incase extraction ambguity)
    root/
        - dataset/
            - train.txt
            - val.txt
            - img/
                - *.jpg
            - cls/
                - *.mat
            - inst/
                - *.mat

        ...
"""
import os
from os import path
from scipy import io as sio
from skimage import io as skio
import numpy as np

import torch
from torch.utils.data import Dataset

DESC_EXT = ".txt"
IMG_EXT = ".jpg"
CLS_EXT = ".mat"
INST_EXT = ".mat"

class SBD(Dataset):
    def __init__(self, root,
            mode= "train", # "train" or "val"
            ann_mode= "inst", # "inst" or "cls"
            max_n_objects= 12, # Due to make a batch of data, the one-hot mask has to be consistent
            sort_objects= True, # NOTE: if sorted, the masks will be arranged in terms of area, not the exact id
        ):
        self._root = root
        self._mode = mode
        self._ann_mode = ann_mode
        self._max_n_objects = max_n_objects
        self._sort_objects = sort_objects

        with open(path.join(self._root, "dataset", mode+".txt")) as f:
            self.filenames = f.read().splitlines()

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def to_onehot(idx_mask, max_n_objects, sort= True):
        """ return a one-hot encoding mask with shape 
        (max(idx_mask)+1, idx_mask.shape[0], idx_mask.shape[1])
        @ Args:
            idx_mask: a 2D array, each pixel's value shows its id
            max_n_objects: it decides the 0-th dimension of the output mask
            sort: if sorted, the masks (excluding background) will be sorted in terms of area of the
                masks, from big to small.
        @ Returns:
            0-th: the stacked masks with shape (N, H, W)
            1-th: a int telling how many instance are in this one-hot mask
        """
        # assign to one-hot
        n_objects_in_mask = np.amax(idx_mask)
        mask_len = np.maximum(n_objects_in_mask, max_n_objects)+1
        masks = [
            np.zeros((idx_mask.shape[0], idx_mask.shape[1]), dtype= np.uint8) \
                for _ in range(mask_len)
        ]
        for i, msk in enumerate(masks):
            msk[idx_mask == i] = 1

        # sort if needed
        if sort:
            masks[1:] = sorted(masks[1:], key= lambda x: x.sum(), reverse= True)
        # chunk if needed
        if len(masks) > (max_n_objects+1):
            masks = masks[:(max_n_objects+1)]
        
        return np.stack(masks), min(n_objects_in_mask, max_n_objects)

    def __getitem__(self, idx):
        image = skio.imread(path.join(self._root, "dataset", "img", self.filenames[idx]+IMG_EXT))
        image = image.astype(np.float32) / 255

        if self._ann_mode == "cls":
            ann = sio.loadmat(path.join(self._root, "dataset","cls",self.filenames[idx]+CLS_EXT))
            ann = ann["GTcls"]
        elif self._ann_mode == "inst":
            ann = sio.loadmat(path.join(self._root, "dataset","inst",self.filenames[idx]+INST_EXT))
            ann = ann["GTinst"]
        else:
            raise ValueError("Wrong annotation mode: {}".format(self._ann_mode))

        # image array pre-processing to (C, H, W) shape
        if len(image.shape) == 2:
            image = np.tile(image, (3,1,1))
        elif len(image.shape) == 3:
            image = image.transpose(2,0,1)
        else:
            raise ValueError("Wrong image shape dimensions for No: {}".format(str(self.filenames[idx])))
        
        # precess masks
        idx_mask = ann["Segmentation"][0][0] # index encoded
        onehot_mask, n_objects = SBD.to_onehot(
            idx_mask,
            max_n_objects= self._max_n_objects,
            sort= self._sort_objects
        )

        return dict(
            image= torch.from_numpy(image), # pixel in [0, 1] scale
            mask= torch.from_numpy(onehot_mask), # NOTE: 0-th dimension of mask is (n_cats+1), 
                # the order of the mas depends on self._supNms or self._catNms
            n_objects= torch.LongTensor([int(n_objects)])[0],
        )

