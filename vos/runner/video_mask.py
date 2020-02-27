from vos.runner.base import RunnerBase
from vos.utils.helpers import overlay_images

from exptools.logging.logger import tf_image_summary

import numpy as np

class VideoMaskRunner(RunnerBase):
    """ A runner that provide basic funtionality of masks videos as demo from `extra_info`
    where `extra_info` consist of following items:
        videos: numpy.ndarray with shape (b, t, C, H, W)
        preds: numpy.ndarray with shape (b, t, n, H, W) with one-hot encoding
    """
    def _store_extra_info(self, itr_i, extra_info, n_select_frames= 1):
        """ For the memory efficiency, this will only randomly choose `n_select_frames` of frames
        to store.
        """
        if not hasattr(self, "_extra_infos"):
            # a hacky way of initialization
            self._extra_infos = []

        videos = extra_info["videos"]
        preds = extra_info["preds"]

        _, T, C, H, W = videos.shape
        _, _, n, _, _ = preds.shape

        # select frames
        t_i = np.random.choice(T, n_select_frames)
        videos = videos[:, t_i]
        preds = preds[:, t_i]

        images = videos.reshape((-1, C, H, W))
        preds = preds.reshape((-1, n, H, W))

        masked_images = overlay_images(images, preds)
        self._extra_infos.extend([image for image in masked_images])

    def _log_extra_info(self, itr_i):
        # transpose from (b, C, H, W) to (b, H, W, C)
        images = np.stack(self._extra_infos, axis= 0).transpose(0,2,3,1)

        # write to summary file
        tf_image_summary("predict masks", data=images, step= itr_i)

        # reset
        del self._extra_infos
        self._extra_infos = []
        