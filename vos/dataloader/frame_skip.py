""" This DataLoader follows protocol of pytorch.utils.data.DataLoader
"""
from random import randint
import re
import torch
from torch.utils.data import DataLoader
from torch._six import container_abcs, string_classes, int_classes

""" Source code copied from https://github.com/pytorch/pytorch/blob/941b42428aaf51489a7e1848bfc145205d372eb7/torch/utils/data/_utils/collate.py#L42
"""

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

class FrameSkipDataLoader(DataLoader):
    """ Generating a function that collate a list of video into torch.Tensor batch.
    NOTE: new dimension will be added
    """
    def __init__(self, dataset,
            n_frames= 3, # num of frames in each item
            skip_frame_range= (0, 25), # a tuple of increasingly max_skip_frames
            skip_increase_interval= 1, # how many rounds of all data before one step if increase.
            **dataloader_kwargs, # other common kwargs that feed to DataLoader
        ):
        self._n_frames = n_frames
        self._skip_frame_range = skip_frame_range
        self._skip_increase_interval = skip_increase_interval
        self._full_dataset_view = 0
        self._choose_skip_length()
        
        def collate_fn(batch):
            """ A modified version of pytorch defualt collate_fn"""

            elem = batch[0]
            elem_type = type(elem)
            if isinstance(elem, torch.Tensor):
                out = None
                # NOTE: different from default collate_fn, it canont avoid memory copy
                return self.stack_videos(batch)
            elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                    and elem_type.__name__ != 'string_':
                elem = batch[0]
                if elem_type.__name__ == 'ndarray':
                    # array of string classes and object
                    if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                        raise TypeError(default_collate_err_msg_format.format(elem.dtype))

                    return collate_fn([torch.as_tensor(b) for b in batch])
                elif elem.shape == ():  # scalars
                    return torch.as_tensor(batch)
            elif isinstance(elem, float):
                return torch.tensor(batch, dtype=torch.float64)
            elif isinstance(elem, int_classes):
                return torch.tensor(batch)
            elif isinstance(elem, string_classes):
                return batch
            elif isinstance(elem, container_abcs.Mapping):
                return {key: collate_fn([d[key] for d in batch]) for key in elem}
            elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
                return elem_type(*(collate_fn(samples) for samples in zip(*batch)))
            elif isinstance(elem, container_abcs.Sequence):
                transposed = zip(*batch)
                return [collate_fn(samples) for samples in transposed]

            raise TypeError(default_collate_err_msg_format.format(elem_type))

        super(FrameSkipDataLoader, self).__init__(dataset, collate_fn= collate_fn, **dataloader_kwargs)

    def _choose_skip_length(self):
        """ Choose a skip length that can be utilized in one data collection
        """
        max_length = min((self._skip_frame_range[1] - self._skip_frame_range[0]), \
            (self._full_dataset_view // self._skip_increase_interval))
        self._skip_length = randint(0, max_length)

    def stack_videos(self, batch):
        """ stack a list of torch.Tensor videos into a batch of data, by sampling from them.
        """
        if len(batch[0].shape < 4):
            return torch.stack(batch, 0)
        # assuming each item is a tensor.Tensor with shape (t, C, H, W)
        samples = []
        for item in batch:
            T = item.shape[0]
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
                    idxs = slice(clip_i, clip_i + self._n_frames*(self._skip_length+1)+1, (self._skip_length+1))    
                samples.extend([item[idxs]])
        return torch.stack(samples, 0)

    def __iter__(self):
        super(FrameSkipDataLoader, self).__iter__()
        self._full_dataset_view += 1
