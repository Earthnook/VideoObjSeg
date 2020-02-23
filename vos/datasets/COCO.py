import torch
import torchvision
from torch.utils import data

from pycocotools.coco import COCO as COCOapi

import os.path as path
import skimage.io as io
import numpy as np

SUBSET_LEN = 50

class COCO(data.Dataset):
    def __init__(self, root,
            mode= "train", # choose between "train", "val"
            is_subset= False, # If is subset, the length will be a fixed small length
        ):
        self._root = root
        self._mode = mode
        self._is_subset = is_subset
        self.coco = COCOapi(
            path.join(self._root, "annotations/instances_{}2017.json".format(self._mode))
        )
        # load categories
        self._cats = self.coco.loadCats(self.coco.getCatIds())
        self._supercats = set([cat['supercategory'] for cat in self._cats])
        self._output_mode = dict(cats= None, is_supcats= False)

        # reset self mode to all categories
        self.set_cats()

    @property
    def all_categories(self):
        return self._cats

    @property
    def all_super_categories(self):
        return self._supercats

    def set_cats(self, cats: list= None, is_supcats= False):
        """ Given category, the dataset will only output all items from that dataset.
        If not provided, all images will be output from __getitem__
        """
        self._output_mode["cats"] = cats
        self._output_mode["is_supcats"] = is_supcats

        if self._output_mode["cats"] is None:
            cats = self._cats
        else:
            cats = self._output_mode["cats"]

        if self._output_mode["is_supcats"]:
            self.imgIds = self.coco.getImgIds(catIds= self.coco.getCatIds(supNms= cats))
        else:
            self.imgIds = self.coco.getImgIds(catIds= self.coco.getCatIds(catNms= cats))

    def __len__(self):
        if self._is_subset:
            return SUBSET_LEN
        else:
            return len(self.imgIds)

    def __getitem__(self, idx):
        if self._is_subset:
            idx = min(SUBSET_LEN, idx)

        img = self.coco.loadImgs(self.imgIds[idx])[0]
        # This image is in (H, W, C) shape
        image = io.imread(img["coco_url"])
        # transpose to (C, H, W) shape
        image = image.transpose(2, 0, 1)

        annIds = self.coco.getAnnIds(imgIds= img["id"])
        anns = self.coco.loadAnns(annIds)

        mask = [self.coco.annToMask(ann) for ann in anns]
        mask = np.array(mask)

        return dict(
            image= image,
            mask= mask,
            anns= anns
        )

if __name__ == "__main__":
    # test code
    import ptvsd
    import sys
    ip_address = ('0.0.0.0', 5050)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush= True)
    # Allow other computers to attach to ptvsd at this IP address and port.
    ptvsd.enable_attach(address=ip_address, redirect_output= True)
    # Pause the program until a remote debugger is attached
    ptvsd.wait_for_attach()
    print("Process attached, start running into experiment...", flush= True)
    ptvsd.break_into_debugger()

    root = sys.argv[1]
    dataset = COCO(root)

    for i in range(len(dataset)):
        x = dataset[i]

    print("debug done...")
        

    