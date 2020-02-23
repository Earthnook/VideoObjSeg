

class FrameSkipDataLoader:
    """ A DataLoader that loads video data, but sample some temporally ordered frames from a video
    as one data item, which makes video length of all items are the same.
    """
    def __init__(self, dataset,
            n_frames= 3, # num of frames in each item
            skip_frame_range= (0, 25), # a tuple of increasingly max_skip_frames
            skip_increase_interval= 1, # how many rounds of all data before one step if increase.
        ):
        self._n_frames = n_frames
        self._skip_frame_range = skip_frame_range
        self._skip_increase_interval = skip_increase_interval
        self._full_dataset_view = 0