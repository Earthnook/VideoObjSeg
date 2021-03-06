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
            max_n_objects= 12, # Due to make a batch of data, the one-hot mask has to be consistent
            sort_anns= False, # Due to multiple anns, sort the anns to get masks as big as possible
        ):
        self._root = root
        self._mode = mode
        self._is_subset = is_subset
        self._max_n_objects = max_n_objects
        self._sort_anns= sort_anns

        self.coco = COCOapi(
            path.join(self._root, "annotations/instances_{}2017.json".format(self._mode))
        )
        # load categories
        self._cats = self.coco.loadCats(self.coco.getCatIds())
        self._catNms = list(set([cat['name'] for cat in self._cats]))
        self._supNms = list(set([cat['supercategory'] for cat in self._cats]))
        self._output_mode = dict(catNms= None, is_supcats= False)

        # reset self mode to all categories
        self.set_cats()

    @property
    def n_objects(self):
        """ NOTE: this method depends on category mode (refer to set_cats).
        And background is also marked as an object
        """
        if self._output_mode["is_supcats"]:
            return len(self._supNms)+1
        else:
            return len(self._catNms)+1

    @property
    def all_categories(self):
        return self._cats

    @property
    def all_categories_names(self):
        return self._catNms

    @property
    def all_super_categories_names(self):
        return self._supNms

    def set_cats(self, cats: list= None, is_supcats= True):
        """ Given category, the dataset will only output all items from that dataset.
        And configure the output mask is in terms of supcats or cats.
        If not provided, all images will be output from __getitem__
        """
        self._output_mode["catNms"] = cats
        self._output_mode["is_supcats"] = is_supcats

        if self._output_mode["catNms"] is None:
            self.imgIds = self.coco.getImgIds()
        else:
            catNms = self._output_mode["catNms"]
            if self._output_mode["is_supcats"]:
                self.imgIds = self.coco.getImgIds(
                    imgIds= self.coco.getImgIds(),
                    catIds= self.coco.getCatIds(supNms= catNms)
                )
            else:
                self.imgIds = self.coco.getImgIds(
                    imgIds= self.coco.getImgIds(),
                    catIds= self.coco.getCatIds(catNms= catNms)
                )

    def __len__(self):
        if self._is_subset:
            return SUBSET_LEN
        else:
            return len(self.imgIds)

    def __getitem__(self, idx):
        if self._is_subset:
            idx = min(SUBSET_LEN, idx)

        try:
            img = self.coco.loadImgs(self.imgIds[idx])[0]
        except:
            raise ValueError("Wrong index: {}".format(idx))
        # This image is in (H, W, C) shape
        image = io.imread('%s/images/%s'%(
            self._root,
            img['file_name']
        )).astype(np.float32)

        annIds = self.coco.getAnnIds(imgIds= img["id"])
        anns = self.coco.loadAnns(annIds)
        if self._sort_anns:
            # put the masks with greatest area at 0-th
            anns.sort(key= lambda x: x["area"], reverse= True)

        # incase of gray-scale image
        if len(image.shape) == 2:
            image = np.tile(image, (3,1,1)).transpose(1,2,0)
        elif len(image.shape) == 3:
            pass
        else:
            raise ValueError("Wrong image shape dimensions\n{}".format(str(img)))
        H, W, _ = image.shape

        # create a bounding box array with (idx0, idx1, H, W) notion.
        # NOTE: it is a bit different from COCO image for the simplicity
        bboxs = np.zeros((self._max_n_objects, 4), dtype= np.float32)
        mask = np.zeros((H, W, self._max_n_objects), dtype= np.uint8)
        bg = np.ones((H, W, 1), dtype= np.uint8) # a background

        n_objects = 0
        for ann_i, ann in enumerate(anns):
            """ Because of the multi-object tracking problem, there is no need to assign to
            specific index.
            """
            if n_objects >= self._max_n_objects: break

            ann_mask = self.coco.annToMask(ann)
            mask[:, :, ann_i] = ann_mask
            bg[:, :, 0] &= (1-ann_mask)

            coco_box = ann["bbox"]
            bboxs[ann_i, 0] = coco_box[1]
            bboxs[ann_i, 1] = coco_box[0]
            bboxs[ann_i, 2] = coco_box[3]
            bboxs[ann_i, 3] = coco_box[2]

            n_objects += 1
        mask = np.concatenate([bg, mask], axis= 2)

        # make the output with dimension order: (C, H, W)
        # NOTE: each data might not have the same resolution
        image = image.transpose(2,0,1).astype(np.float32) / 255
        mask = mask.transpose(2,0,1).astype(np.uint8)
        return dict(
            image= torch.from_numpy(image), # pixel in [0, 1] scale
            mask= torch.from_numpy(mask), # NOTE: 0-th dimension of mask is (n_cats+1), 
                # the order of the mas depends on self._supNms or self._catNms
            n_objects= torch.tensor(n_objects),
        )

if __name__ == "__main__":
    # test code
    import ptvsd
    import sys
    # ip_address = ('0.0.0.0', 5050)
    # print("Process: " + " ".join(sys.argv[:]))
    # print("Is waiting for attach at address: %s:%d" % ip_address, flush= True)
    # # Allow other computers to attach to ptvsd at this IP address and port.
    # ptvsd.enable_attach(address=ip_address, redirect_output= True)
    # # Pause the program until a remote debugger is attached
    # ptvsd.wait_for_attach()
    # print("Process attached, start running into experiment...", flush= True)
    # ptvsd.break_into_debugger()

    root = sys.argv[1]
    dataset = COCO(root)

    dataloader = data.DataLoader(dataset,
        batch_size=128, 
        shuffle= True, 
        num_workers= 48
    )

    for i, b in enumerate(dataloader):
        print("Get a batch, {}: type({})".format(i, type(b)))

    print("debug done...")
        

    