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
    def __init__(self, **kwargs):
        super(VideoMaskRunner, self).__init__(**kwargs)
        self._extra_infos = None
        self._eval_extra_infos = None

    def _store_extra_info(self, itr_i, extra_info, n_select_frames= 1, evaluate= False):
        """ For the memory efficiency, this will only randomly choose `n_select_frames` of frames
        to store.
        """
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

        if not evaluate:
            if self._extra_infos is None:
                # a hacky way of initialization
                self._extra_infos = {k: list() for k in ["images", "masks", "preds"]}
            self._extra_infos["images"].extend([image for image in images])
            self._extra_infos["masks"].extend([mask for mask in masks])
            self._extra_infos["preds"].extend([pred for pred in preds])
        else:
            if self._eval_extra_infos is None:
                # a hacky way of initialization
                self._eval_extra_infos = {k: list() for k in ["images", "masks", "preds"]}
            self._eval_extra_infos["images"].extend([image for image in images])
            self._eval_extra_infos["masks"].extend([mask for mask in masks])
            self._eval_extra_infos["preds"].extend([pred for pred in preds])

    def _log_extra_info(self, itr_i, n_select_samples= 4, evaluate= False):
        # transpose from (b, C, H, W) to (1, H, b*W, C)
        if evaluate and not self._eval_extra_infos is None:
            images = np.stack(self._eval_extra_infos["images"], axis= 0)
            masks = np.stack(self._eval_extra_infos["masks"], axis= 0)
            preds = np.stack(self._eval_extra_infos["preds"], axis= 0)
        elif not evaluate and not self._extra_infos is None:
            images = np.stack(self._extra_infos["images"], axis= 0)
            masks = np.stack(self._extra_infos["masks"], axis= 0)
            preds = np.stack(self._extra_infos["preds"], axis= 0)
        else:
            return
        
        B, C, H, W = images.shape
        _, N, _, _ = masks.shape
        b_i = np.random.choice(B, n_select_samples)
        s_images = stack_images(images[b_i]) # (1, C, H, b*W)
        s_masks = stack_masks(masks[b_i]) # (1, 1, b*H, N*W)
        s_preds = stack_masks(preds[b_i])

        # write to summary file
        tf_image_summary(("Eval " if evaluate else "") + "input images",
            data= s_images.transpose(0,2,3,1),
            step= itr_i
        )
        tf_image_summary(("Eval " if evaluate else "") + "ground truths",
            data= s_masks.transpose(0,2,3,1) * 255,
            step= itr_i
        )
        tf_image_summary(("Eval " if evaluate else "") + "predictions",
            data= s_preds.transpose(0,2,3,1) * 255,
            step= itr_i
        )

        # reset
        if evaluate:
            del self._eval_extra_infos
            self._eval_extra_infos = None
        else:
            del self._extra_infos
            self._extra_infos = None
        
    # def log_data_info(self, itr_i, data, n_select_frames= 1):
    #     """ In case of data pre-processing bugs, log images into tensorflow
    #     each image should be in batch
    #     """
    #     videos = data["video"].cpu().numpy()
    #     masks = data["mask"].cpu().numpy()
    #     n_objects = data["n_objects"].cpu().numpy()

    #     b, T, C, H, W = videos.shape
    #     _, _, N, _, _ = masks.shape
    #     # select frames
    #     t_i = np.random.choice(T, n_select_frames)
    #     images = videos[:, t_i].reshape(-1, C, H, W) # (b*t, C, H, W)
    #     masks = masks[:, t_i].reshape(-1, N, H, W) # (b*t, N, H, W)
    #     s_images = stack_images(images)
    #     s_masks = stack_masks(masks) # (1, 1, b*t*H, N*W)

    #     tf_image_summary("data images",
    #         data= s_images.transpose(0,2,3,1),
    #         step= itr_i
    #     )
    #     tf_image_summary("data masks",
    #         data=s_masks.transpose(0,2,3,1) * 255, # int are treated as 0-255 scale
    #         step= itr_i,
    #     )
