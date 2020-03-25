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

import cv2
import numpy as np

to_tensor = ToTensor()

def contour_mask_2_mask(contour_mask):
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(contour_mask)
    mask = cv2.drawContours(mask, contours, -1, 1, -1)
    return mask

class VOCSegmentation(datasets.VOCSegmentation):
    """
    NOTE: This dataset has only one segmentation at a time
    """
    def __init__(self, root,
            mode= "train", # "train", "val" or "trainval"
            year= "2012", # from "2007" to "2012"
            max_n_objects= 1, # decides the channel of masks (in order to make batch)
            sort_masks= True, # if multiple targets (seems unlikely), sort to put biggest at 0th
        ):
        super().__init__(root, year= year, image_set= mode)
        self.max_n_objects = max_n_objects
        self.sort_masks = sort_masks

    def __getitem__(self, idx):
        image, targets = super().__getitem__(idx)
        image = to_tensor(image) # (C, H, W)
        targets = to_tensor(targets).numpy() # (N, H, W)
        targets = targets.astype(np.uint8) # (H, W) with 0,1 incode

        bg = np.ones_like(targets[0])
        masks = []
        n_objects = 0
        for target in targets:
            n_objects += 1
            mask = contour_mask_2_mask(target)
            bg &= 1-mask
            masks.append(mask)
            if n_objects >= self.max_n_objects:
                break
        if self.sort_masks:
            masks.sort(key= lambda x: np.sum(x), reverse= True)

        masks = np.stack([bg]+masks)
        return dict(
            image= image,
            mask= torch.from_numpy(masks),
            n_objects= torch.tensor(1),
        )
