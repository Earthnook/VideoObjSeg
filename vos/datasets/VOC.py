"""
dataset interface from https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.VOCSegmentation
assuming the root directory is like:
    root/
        - VOCdevkit/
            - VOC{year}/
                ...

"""
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

import os
import cv2
import numpy as np

to_tensor = ToTensor()

class VOCSegmentationObject(datasets.VOCSegmentation):
    """ Modified from pytorch source code, in order to make instance-level segmentation
    """

    def __init__(self,
                 root,
                 year='2012',
                 image_set='train',
                 **kwargs):
        super(VOCSegmentationObject, self).__init__(
            root= root,
            year= year,
            image_set= image_set,
            **kwargs
        )
        valid_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_sets.append("test")
        base_dir = datasets.voc.DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        mask_dir = os.path.join(voc_root, 'SegmentationObject')
        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

class VOCSegmentation(VOCSegmentationObject):
    """
    NOTE: This dataset has only one segmentation at a time
    """
    def __init__(self, root,
            mode= "train", # "train", "val" or "trainval"
            year= "2012", # from "2007" to "2012"
            max_n_objects= 4, # decides the channel of masks (in order to make batch)
            sort_masks= True, # if multiple targets (seems unlikely), sort to put biggest at 0th
        ):
        super().__init__(root, year= year, image_set= mode)
        self.max_n_objects = max_n_objects
        self.sort_masks = sort_masks

    @staticmethod
    def make_instance_mask(contour_mask):
        """ given VOC targets (a uint8 np array) ranging [0, 255] output one-hot masks as instances
        NOTE: the object order is not placed in classes or instance id
        """
        hist, _ = np.histogram(contour_mask.reshape(-1), bins= 256)
        values = np.nonzero(hist)[0] # it is sure to be a len 1 tuple
        # build masks
        fgs = list()
        for value in values[1:-1]: # exclude 0 and 255
            fgs.append((contour_mask == value).astype(np.uint8))
        return fgs # len(fgs) == n_objects

    def __getitem__(self, idx):
        image, targets = super().__getitem__(idx)
        image = to_tensor(image) # (C, H, W)
        plt.imshow(targets); plt.figure()
        targets = to_tensor(targets).numpy()*255 # (N, H, W)

        print(self.masks[idx])
        plt.hist(targets[0][(targets[0] > 0) & (targets[0] < 255)].reshape(-1), bins= int(256))
        plt.show()
        plt.figure()
        plt.imshow((targets[0] > 0) & (targets[0] < 255))
        print(((targets[0] > 0) & (targets[0] < 255)).max())

        targets = targets.astype(np.uint8)[0] # (H, W) with [0,255] encode
        fgs = VOCSegmentation.make_instance_mask(targets)
        n_objects = len(fgs)

        bg = np.ones_like(targets)
        if self.sort_masks:
            fgs.sort(key= lambda x: np.sum(x), reverse= True)
        if n_objects >= self.max_n_objects:
            fgs = fgs[:self.max_n_objects]
            n_objects = self.max_n_objects
        elif n_objects < self.max_n_objects:
            fgs.extend([
                np.zeros_like(targets) for _ in range(self.max_n_objects - n_objects)
            ])
        for fg in fgs:
            bg &= 1-fg

        masks = np.stack([bg]+fgs) # (max_n_objects+1, H, W)
        return dict(
            image= image,
            mask= torch.from_numpy(masks),
            n_objects= torch.tensor(n_objects),
        )
