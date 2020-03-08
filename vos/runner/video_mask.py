from vos.runner.base import RunnerBase
from vos.utils.helpers import stack_images, stack_masks

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
        if not hasattr(self, "_extra_infos") or self._extra_infos is None:
            # a hacky way of initialization
            self._extra_infos = {k: list() for k in ["images", "masks", "preds"]}

        videos = extra_info["videos"]
        masks = extra_info["masks"]
        preds = extra_info["preds"]

        B, T, C, H, W = videos.shape
        _, _, N, _, _ = masks.shape
        _, _, n, _, _ = preds.shape

        # select frames
        t_i = np.random.choice(T, n_select_frames)
        images = videos[:, t_i]
        masks = masks[:, t_i]
        preds = preds[:, t_i]

        images = images.reshape((-1, C, H, W))
        masks = masks.reshape((-1, N, H, W))
        preds = preds.reshape((-1, n, H, W)) # (1, C, H, B*W)

        self._extra_infos["images"].extend([image for image in images])
        self._extra_infos["masks"].extend([mask for mask in masks])
        self._extra_infos["preds"].extend([pred for pred in preds])

    def _log_extra_info(self, itr_i):
        # transpose from (b, C, H, W) to (1, H, b*W, C)
        images = np.stack(self._extra_infos["images"], axis= 0)
        masks = np.stack(self._extra_infos["masks"], axis= 0)
        preds = np.stack(self._extra_infos["preds"], axis= 0)
        s_images = stack_images(images) # (1, C, H, b*W)
        s_masks = stack_masks(masks) # (1, 1, b*H, N*W)
        s_preds = stack_masks(preds)

        # write to summary file
        tf_image_summary("input images",
            data= s_images.transpose(0,2,3,1),
            step= itr_i
        )
        tf_image_summary("ground truths",
            data= s_masks.transpose(0,2,3,1) * 255,
            step= itr_i
        )
        tf_image_summary("predictions",
            data= s_preds.transpose(0,2,3,1) * 255,
            step= itr_i
        )

        # reset
        del self._extra_infos
        self._extra_infos = None
        
    def log_data_info(self, itr_i, data, n_select_frames= 1):
        """ In case of data pre-processing bugs, log images into tensorflow
        each image should be in batch
        """
        videos = data["video"].cpu().numpy()
        masks = data["mask"].cpu().numpy()
        n_objects = data["n_objects"].cpu().numpy()

        b, T, C, H, W = videos.shape
        _, _, N, _, _ = masks.shape
        # select frames
        t_i = np.random.choice(T, n_select_frames)
        images = videos[:, t_i].reshape(-1, C, H, W) # (b*t, C, H, W)
        masks = masks[:, t_i].reshape(-1, N, H, W) # (b*t, N, H, W)
        s_images = stack_images(images)
        s_masks = stack_masks(masks) # (1, 1, b*t*H, N*W)

        tf_image_summary("data images",
            data= s_images.transpose(0,2,3,1),
            step= itr_i
        )
        tf_image_summary("data masks",
            data=s_masks.transpose(0,2,3,1) * 255, # int are treated as 0-255 scale
            step= itr_i,
        )
