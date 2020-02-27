import torch
from torch.utils.data import Dataset

from numpy.random import randint

from vos.utils.quick_args import save__init__args

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
        ):
        save__init__args(locals(), underscore= True)
        # As curriculum learning, these surve as counters
        self._num_getitems = 0
        self._full_dataset_view = 0
        self._choose_skip_length()

    def __len__(self):
        return self._max_clips_sample * self._dataset.__len__()

    def __getitem__(self, idx):
        idx = int(idx // self._max_clips_sample)
        item = self._dataset.__getitem__(idx)
        _, video = self.stack_video(item["video"])
        b, mask = self.stack_video(item["mask"])

        # record and reset
        self._num_getitems += 1
        if self._num_getitems == self.__len__():
            self._full_dataset_view += 1
            self._num_getitems = 0
        self._choose_skip_length()
        
        return dict(
            video= video,
            mask= mask,
            n_objects= item["n_objects"].expand(b,)
        )

    def _choose_skip_length(self):
        """ Choose a skip length that can be utilized in one data collection
        """
        max_length = min((self._skip_frame_range[1] - self._skip_frame_range[0]), \
            (self._full_dataset_view // self._skip_increase_interval))
        self._skip_length = 0 if max_length == 0 else randint(0, max_length)

    def stack_video(self, video):
        """ stack a list of torch.Tensor videos into a batch of data, by sampling from them.
        """
        # assuming each item is a tensor.Tensor with shape (t, C, H, W)
        samples = []
        T = video.shape[0]
            # +-----------------------------------+
            # |... ... ... ...     F    F    F    F
            # +--------------------+--------------+
            #                      |    A clip    |
            #                      +--------------+
        n_clips = T - (self._n_frames-1) * (self._skip_length+1)
        for clip_i in range(n_clips):
            if self._n_frames == 1:
                idxs = slice(clip_i, clip_i+1)
            else:
                idxs = slice(clip_i, clip_i + self._n_frames*(self._skip_length+1), (self._skip_length+1))
            samples.extend([video[idxs]])
            if clip_i+1 >= self._max_clips_sample: break
        b = len(samples)
        return b, torch.stack(samples, 0) # (b, t, C, H, W)

    @staticmethod
    def collate_fn(batch: list):
        """ Do provide this function to the dataloader, 
        argument name is the same as this function name
        """
        b_ = {k: list() for k in batch[0].keys()}
        for item in batch:
            for k in item.keys():
                b_[k].extend([item[k]])

        for k in b_.keys():
            b_[k] = torch.cat(b_[k], dim= 0).contiguous()

        return b_