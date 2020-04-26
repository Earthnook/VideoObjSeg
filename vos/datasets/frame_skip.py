import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from numpy.random import randint

from vos.utils.quick_args import save__init__args
from vos.utils.image_shaper import random_crop

class FrameSkipDataset(Dataset):
    """ This is a wrapper that designed as firstly STM requested to sample training frames.
    And you should be sure that the given video dataset should be competible during testing.

        NOTE: The output of __len__ method does not match the exact sum of the batch size.
    """
    def __init__(self, dataset,
            n_frames= 3, # num of frames in each item
            skip_frame_range= (0, 25), # a tuple of increasingly max_skip_frames
            skip_increase_interval= 1, # how many rounds of all data before one step if increase.
            max_clips_sample= 2, # Due memory limitation, sample all clips from one video might 
                # even explode all the cuda memory.
            resolution= (384, 384), # control the output of the image, to make a batch
            resize_method= "crop", # choose between "crop", "interpolate"
            update_on_full_view= True, # when update on full view, skip interval will only be increase when all data is viewed once
        ):
        save__init__args(locals(), underscore= True)
        # As curriculum learning, these surve as counters
        self._num_getitems = 0
        self._full_dataset_view = 0
        self._choose_skip_length()

    def _choose_skip_length(self):
        """ update self._skip_length attribute, Choose a skip length that can be utilized in one data collection
        """ 
        if self._update_on_full_view:
            max_length = self._full_dataset_view // self._skip_increase_interval
        else:
            max_length = self._num_getitems // self._skip_increase_interval
        max_length = min((self._skip_frame_range[1] - self._skip_frame_range[0]), max_length) + self._skip_frame_range[0]

        self._skip_length = 0 if max_length == 0 else randint(0, max_length)

    def clip_video(self, video, idx):
        """ clip a torch.Tensor video by sampling from them.
        """
        # assuming each item is a tensor.Tensor with shape (t, C, H, W)
        T = video.shape[0]
            # +-----------------------------------+
            # |... ... ... ...     F    F    F    F
            # +--------------------+--------------+
            #                      |    A clip    |
            #                      +--------------+
        # some pre-checking work
        if T < self._n_frames:
            raise ValueError("Too short video source, please check video, shape: {}".format(video.shape))
        total_skipped_length = (self._n_frames-1) * (self._skip_length+1) + 1
        while T < total_skipped_length:
            self._skip_length -= 1
            total_skipped_length = (self._n_frames-1) * (self._skip_length+1) + 1
        n_clips = T - total_skipped_length + 1

        clip_i = min(idx, n_clips-1)
        if self._n_frames == 1:
            idxs = slice(clip_i, clip_i+1)
        else:
            idxs = slice(clip_i, clip_i + self._n_frames*(self._skip_length+1), (self._skip_length+1))
        return_ =  video[idxs] # (t, C, H, W)
        assert return_.shape[0] == self._n_frames, f"Wrong clip shape: {return_.shape}, video shape: {video.shape}, skip_length: {self._skip_length}, n_frames: {self._n_frames}"
        return return_

    def __len__(self):
        return self._max_clips_sample * self._dataset.__len__()

    @staticmethod
    def interpo(resolution, *videos):
        """ run nn.functional.interpolate
        @ Args
            resolution: a tuple of 2 ints
            videos: a sequence of tensor with shape (c, H, W)
        @ Returns
            return_: a sequence of tensor with shape (c, *resolution)
        """
        return_ = []
        for v in videos:
            v = v.to(dtype= torch.float32)
            return_.append(F.interpolate(v, resolution))
        return return_
    
    def __getitem__(self, idx):
        video_idx = int(idx // self._max_clips_sample)
        sub_idx = int(idx % self._max_clips_sample)
        item = self._dataset.__getitem__(video_idx)

        self._choose_skip_length()
        video = self.clip_video(item["video"], sub_idx)
        mask = self.clip_video(item["mask"], sub_idx)

        if self._resize_method == "crop":
            video, mask = random_crop(self._resolution, video, mask)
        elif self._resize_method == "interpolate":
            video, mask = self.interpo(self.resolution, video, mask)
            mask = mask.to(dtype= torch.uint8)
        else:
            raise NotImplementedError

        # record and reset
        self._num_getitems += 1
        if self._num_getitems >= self.__len__():
            self._full_dataset_view += 1
            self._num_getitems = 0
        
        return dict(
            video= video,
            mask= mask,
            n_objects= item["n_objects"]
        )