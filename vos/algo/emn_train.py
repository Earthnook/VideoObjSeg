from vos.algo.videoobjseg import VideoObjSegAlgo
from vos.utils.helpers import pad_divide_by

import numpy as np
import torch
from torch.nn import functional as F

def build_target_image(image, mask, n_objects):
    """ 
    @ Args:
        image: shape with (b, C, H, W) for one batch-of frame
        mask: shape with (b, N, H, W) where first `n_objects` channels in each data should be the \
mask for the objects
        n_objects: shape with (b,) telling how many object is in each data `n_objects[i] <= N+1`
    @ Returns:
        target_image: shape with (b, n_objects, C, h', w') where the target image for each object \
could be smaller.
    """
    with torch.no_grad():
        n_objects = max(n_objects.max().item(), 1)
        B, N, H, W = mask.shape
        _, C, _, _ = image.shape
        assert N >= n_objects+1

        img_e = image.unsqueeze(1).expand(B, n_objects, -1,-1,-1) # (B, no, C, H, W)
        mask_e = mask[:, 1:n_objects+1].unsqueeze(2).expand(-1,-1, C, -1,-1) # (B, no, C, H, W)

        target_images = img_e * mask_e
    b, no, c, h, w = target_images.shape
    img = target_images.view(-1, c, h, w)
    [img], _ = pad_divide_by([img], 16, (h, w))
    H, W = img.shape[-2:]
    target_images = img.view(b, no, c, H, W)
    return target_images


class EMNAlgo(VideoObjSegAlgo):
    """ The algorithm re-producing Enhanced Memory Network at 
    https://youtube-vos.org/assets/challenge/2019/reports/YouTube-VOS-01_Enhanced_Memory_Network_for_Video_Segmentation.pdf
    """
    def step(self,
            frames: torch.Tensor,
            masks: torch.Tensor,
            n_objects: int,
            Mem_every=None, Mem_number=None,
        ):
        """ go through the data and calculate the loss (torch.Variable)
        And all arrays in this function are torch.Tensors
        @ Args:
            frames: batch-wise, shape (b, t, C, H, W)
            masks: batch-wise, shape (b, t, n, H, W)
            n_objects: A int telling how many objects (max) in this batch of data
        """
        n_frames = frames.size()[1]
        # initialize storage tensors
        if Mem_every:
            to_memorize = [int(i) for i in np.arange(0, n_frames, step=Mem_every)]
        elif Mem_number:
            to_memorize = [int(round(i)) for i in np.linspace(0, n_frames, num=Mem_number+2)[:-1]]
        else:
            raise NotImplementedError

        estimates = torch.zeros_like(masks, dtype= torch.float32)
        logits = torch.zeros_like(masks, dtype= torch.float32)
        estimates[:, 0] = masks[:, 0]

        target_images = build_target_image(frames[:, 0], masks[:, 0], n_objects)

        for t in range(1, n_frames):
            # memorize
            prev_key, prev_value = self.model(
                frames[:, t-1],
                estimates[:, t-1],
                n_objects,
            )

            if t-1 == 0:
                this_keys, this_values = prev_key, prev_value
            else:
                this_keys = torch.cat([keys, prev_key], dim= 3)
                this_values = torch.cat([values, prev_value], dim= 3)

            # segment
            logit = self.model(
                frames[:, t],
                this_keys,
                this_values,
                n_objects,
                target_images,
            )
            logits[:, t] = logit
            estimates[:, t] = F.softmax(logit, dim= 1)

            # update memory
            if t-1 in to_memorize:
                keys, values = this_keys, this_values

        pred = (estimates.detach() == estimates.detach().max(dim= 2, keepdim= True)[0])
        pred = pred.to(dtype= torch.uint8)
        return pred, self.loss_fn(logits[:, 1:], masks[:, 1:]) # only query frames are used
