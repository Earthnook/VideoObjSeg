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

to_tensor = ToTensor()

class VOCSegmentation(datasets.VOCSegmentation):
    """
    NOTE: This dataset has only one segmentation at a time
    """
    def __getitem__(self, idx):
        image, target = super().__getitem__(idx)
        image = to_tensor(image) # (C, H, W)
        target = to_tensor(target)[0]
        target = target.to(dtype= torch.uint8) # (H, W) with 0,1 incode

        mask = torch.stack((
            1-target, target,
        ))
        return dict(
            image= image,
            mask= mask,
            n_objects= torch.tensor(1),
        )
