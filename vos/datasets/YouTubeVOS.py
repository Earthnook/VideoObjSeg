"""
Self implmented YouTube VOS 2019 dataset, support "train" and "val". "test" has not been tested.
The expected structure is
    root/
        - JPEGImages/
            - {video_id}/
                - {frame_id}.jpg
        - Annotations/
            - {video_id}/
                - {frame_id}.png
        - meta.json
"""
import torch
from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt # for debugging

class YouTubeVOS(Dataset):
    """ NOTE: This dataset will prevent objects that appears in the middle of the video
        NOTE: Don't give validation dataset because they don't have ground-truths except
        for init_frames, which is not compatible for this implementation
    """
    def __init__(self, root,
            max_n_objects= 4, # this configures the channel dimension of masks
        ):
        self.root = root
        self.max_n_objects = max_n_objects
        self.video_ids = os.listdir(os.path.join(self.root, "Annotations"))
        self.video_ids.sort()

    def __len__(self):
        return len(self.video_ids)

    @staticmethod
    def to_onehot(idx_mask, object_ids, max_n):
        """ Given masks in index representation, returns a one-hot representation
        NOTE: Due to it is video, area of objects might change, this function does
        not support sorting based on area.
        @ Args:
            idx_mask: array with shape (T, H, W)
            object_ids: array with shape (N,) and does not include background
        @ Returns:
            0th: array with shape (T, max_n+1, H, W) with background done
        """
        T, H, W = idx_mask.shape
        bg = np.ones((T, H, W), dtype= np.uint8)
        fgs = np.zeros((T, max_n, H, W), dtype= np.uint8)
        for i, obj_id in enumerate(object_ids):
            frame_layer = fgs[:, i] # (T, H, W) with same memory
            frame_layer[idx_mask == obj_id] = 1
            bg &= 1-frame_layer
        bg = np.expand_dims(bg, axis= 1)
        return np.concatenate([bg, fgs], axis= 1)

    def __getitem__(self, idx):
        assert idx < len(self), "Invalid idx: {}".format(idx)
        video_id = self.video_ids[idx]
        frame_names = os.listdir(os.path.join(self.root, "JPEGImages", video_id))
        frame_names.sort()
        masks_names = os.listdir(os.path.join(self.root, "Annotations", video_id))
        masks_names.sort()
        assert len(frame_names) == len(masks_names), \
            "Wrong data pair: {}, len(frames): {}, len(masks): {}".format(
                video_id, len(frame_names), len(masks_names)
            )

        T = len(frame_names)
        init_mask = Image.open(os.path.join(self.root, "Annotations", video_id, masks_names[0]))
        init_mask = np.array(init_mask.convert("P")) # (H, W)
        init_frame = Image.open(os.path.join(self.root, "JPEGImages", video_id, frame_names[0]))
        init_frame = np.array(init_frame.convert("RGB")) / 255 # (H, W, C)
        H, W, C = init_frame.shape

        object_ids = np.unique(init_mask, return_counts= False)[1:]
        # by setting return_counts= True, you might can sort masks by area
        if len(object_ids) > (self.max_n_objects):
            object_ids = object_ids[:self.max_n_objects]

        frames = np.zeros((T, H, W, C), dtype= np.float32)
        frames[0] = init_frame
        for i in range(1, T):
            file = os.path.join(self.root, "JPEGImages", video_id, frame_names[i])
            frame = np.array(Image.open(file).convert("RGB")) / 255
            frames[i] = frame
        frames = frames.transpose(0,3,1,2) # shape (T, C, H, W)

        idx_masks = np.zeros((T, H, W), dtype= np.uint8)
        idx_masks[0] = init_mask
        for i in range(1, T):
            file = os.path.join(self.root, "Annotations", video_id, masks_names[i])
            mask = np.array(Image.open(file).convert("P"), dtype= np.uint8)
            idx_masks[i] = mask
        
        masks = YouTubeVOS.to_onehot(idx_masks, object_ids, self.max_n_objects)

        return dict(
            video= torch.from_numpy(frames),
            mask= torch.from_numpy(masks),
            n_objects= torch.LongTensor([len(object_ids)])[0],
        )