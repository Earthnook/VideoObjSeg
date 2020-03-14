from vos.algo.videoobjseg import VideoObjSegAlgo
from vos.utils.helpers import extract_bboxs

import numpy as np
import torch
from torch.nn import functional as F

def mask_targets(image, mask, n_objects):
    """ 
    @ Args:
        image: shape with (b, C, H, W) for one batch-of frame
        mask: shape with (b, N, H, W) in one-hot encoding and N >= n_objects+1
        n_objects: shape with (b,) telling how many object is in each data `n_objects[i] <= N+1`
    @ Returns:
        target_image: shape with (b, n_objects, C, H, W)
    """
    # mask the images
    with torch.no_grad():
        n_objects = max(n_objects.max().item(), 1)
        B, N, H, W = mask.shape
        _, C, _, _ = image.shape
        assert N >= n_objects+1

        img_e = image.unsqueeze(1).expand(B, n_objects, -1,-1,-1) # (B, no, C, H, W)
        mask_e = mask[:, 1:n_objects+1].unsqueeze(2).expand(-1,-1, C, -1,-1) # (B, no, C, H, W)

        target_images = img_e * mask_e.type_as(img_e)
    return target_images

def extract_targets(images, mask, n_objects):
    """ NOTE: in some cases, maybe all bounding boxes are in shape (0,0) this method will return a
    (16, 16) patch, which is likely be a full-black patch.
    @ Args:
        images: torch.Tensor with shape (b, n_objects, C, H, W)
        mask: torch.Tensor with shape (b, N, H, W) where N >= n_objects+1
    @ Returns:
        targets: torch.Tensor with shape (b, n_objects, C, h', w')
    """
    o_mask = mask[:,1:]
    b, N, H, W = o_mask.shape
    C = images.shape[2]
    with torch.no_grad():
        n_objects = max(n_objects.max().item(), 1)
        bboxs = extract_bboxs(o_mask.reshape((-1, H, W)).numpy()).reshape((b, N, 4))
    maxH = int(bboxs[:,:,2].max())
    maxW = int(bboxs[:,:,3].max())
    # incase no object is extracted, aka. (H, W) == (0, 0)
    maxH = max(32, maxH)
    maxW = max(32, maxW)

    targets = torch.zeros((b, n_objects, C, maxH, maxW))

    # Sorry, I can only think of process them one by one
    for b_i in range(b):
        for n_i in range(n_objects):
            Hidx, Widx, Hlen, Wlen = bboxs[b_i, n_i]
            tar_image = images[b_i,n_i,:, Hidx:Hidx+Hlen, Widx:Widx+Wlen]
            padding = (0,maxW-Wlen,0,maxH-Hlen)
            targets[b_i, n_i] = F.pad(tar_image, pad= padding)
    
    return targets


class EMNAlgo(VideoObjSegAlgo):
    """ The algorithm re-producing Enhanced Memory Network at 
    https://youtube-vos.org/assets/challenge/2019/reports/YouTube-VOS-01_Enhanced_Memory_Network_for_Video_Segmentation.pdf
    """
    def step(self,
            frames: torch.Tensor,
            masks: torch.Tensor,
            n_objects: torch.Tensor,
            Mem_every=None, Mem_number=None,
        ):
        """ go through the data and calculate the loss (torch.Variable)
        And all arrays in this function are torch.Tensors
        @ Args:
            frames: batch-wise, shape (b, t, C, H, W)
            masks: batch-wise, shape (b, t, n, H, W)
            n_objects: A batch-wise int telling how many objects in this batch of data
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

        targets = mask_targets(frames[:, 0], masks[:, 0], n_objects)
        targets = extract_targets(targets, masks[:, 0], n_objects)

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
                targets
            )
            logits[:, t] = logit
            estimates[:, t] = F.softmax(logit, dim= 1)

            # update memory
            if t-1 in to_memorize:
                keys, values = this_keys, this_values

        pred = (estimates.detach() == estimates.detach().max(dim= 2, keepdim= True)[0])
        pred = pred.to(dtype= torch.uint8)

        # select loss calculation
        loss_idx = 0 if self.include_bg_loss else 1
        b, t, n, H, W = masks.shape
        p = estimates[:,1:,loss_idx:].reshape(-1, H, W)
        g = masks[:,1:,loss_idx:].reshape(-1, H, W).to(dtype= torch.float32)

        # calculate loss and return
        return pred, self.loss_fn(p, g) # only query frames are used
