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
        if not hasattr(self, "_extra_infos") or self._extra_infos is None:
            # a hacky way of initialization
            self._extra_infos = {k: list() for k in ["images", "masks", "preds"]}

        videos = extra_info["videos"]
        masks = extra_info["masks"]
        preds = extra_info["preds"]

        _, T, C, H, W = videos.shape
        _, _, N, _, _ = masks.shape
        _, _, n, _, _ = preds.shape

        # select frames
        t_i = np.random.choice(T, n_select_frames)
        images = videos[:, t_i]
        masks = masks[:, t_i]
        preds = preds[:, t_i]

        images = videos.reshape((-1, C, H, W))
        masks = masks.reshape((-1, N, H, W))
        preds = preds.reshape((-1, n, H, W))

        self._extra_infos["images"].extend([image for image in images])
        self._extra_infos["masks"].extend([image for image in masks])
        self._extra_infos["preds"].extend([image for image in preds])

    def _log_extra_info(self, itr_i):
        # transpose from (b, C, H, W) to (b, H, W, C)
        images = np.stack(self._extra_infos["images"], axis= 0).transpose(0,2,3,1)
        # due to multi-channel, masks and preds are a bit different
        masks = [np.expand_dims(np.hstack(x[:]), axis= 2) \
            for x in self._extra_infos["masks"]] # a list of (H, n*W, 1)
        preds = [np.expand_dims(np.hstack(x[:]), axis= 2) \
            for x in self._extra_infos["preds"]] # a list of (H, n*W, 1)

        # write to summary file
        tf_image_summary("input images", data=images, step= itr_i)
        for pred in preds:
            tf_image_summary("ground truths",
                data= np.expand_dims(masks, axis= 0) * 255,
                step= itr_i
            )
        for pred in preds:
            tf_image_summary("predictions",
                data= np.expand_dims(pred, axis= 0) * 255,
                step= itr_i
            )

        # reset
        del self._extra_infos
        self._extra_infos = None
        
    def log_data_info(self, itr_i, data):
        """ In case of data pre-processing bugs, log images into tensorflow
        each image should be in batch
        """
        videos = data["video"].cpu().numpy()
        masks = data["mask"].cpu().numpy()
        n_objects = data["n_objects"].cpu().numpy()

        _, T, C, H, W = videos.shape
        _, _, n, _, _ = masks.shape
        images = videos.reshape((-1, C, H, W))
        masks_1 = masks.reshape((-1, n, H, W))[:, 1:2] # choose only the first object to track
        bg = masks.reshape((-1, n, H, W))[:, 0:1] # choose the background mask
        masked_images = overlay_images(images, masks_1, alpha= 0.2)
        
        tf_image_summary("data images", data= images.transpose(0,2,3,1), step= itr_i)
        tf_image_summary("data images with mask", 
            data=masked_images.transpose(0,2,3,1),
            step= itr_i,
        )
        tf_image_summary("data masks (first object)",
            data=masks_1.transpose(0,2,3,1) * 255, # int are treated as 0-255 scale
            step= itr_i,
        )
        tf_image_summary("data masks (background)",
            data= bg.transpose(0,2,3,1) * 255,
            step= itr_i,
        )
