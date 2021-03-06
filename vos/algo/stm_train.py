from vos.algo.videoobjseg import VideoObjSegAlgo

import numpy as np
import torch
from torch.nn import functional as F

class STMAlgo(VideoObjSegAlgo):
    """ The algorithm re-producing STM training method
    https://arxiv.org/abs/1904.006071
    """
    def step(self,
            frames: torch.Tensor,
            masks: torch.Tensor,
            n_objects: int,
            calc_loss= True,
            Mem_every=None, Mem_number=None,
        ):
        """ go through the data and calculate the loss (torch.Variable)
        And all arrays in this function are torch.Tensors
        @ Args:
            frames: batch-wise, shape (b, t, C, H, W)
            masks: batch-wise, shape (b, t, n, H, W)
            n_objects: batch-wise int telling how many objects in among batch of data
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
            )
            logits[:, t] = logit
            estimates[:, t] = F.softmax(logit, dim= 1)

            # update memory
            if t-1 in to_memorize:
                keys, values = this_keys, this_values

        pred = (estimates.detach() == estimates.detach().max(dim= 2, keepdim= True)[0])
        pred = pred.to(dtype= torch.uint8)

        p = estimates[:,1:]
        g = masks[:,1:].to(dtype= torch.float32)

        # calculate loss and return
        loss = self.loss_fn(p, g, n_objects) if calc_loss else 0 # only query frames are used
        return pred, loss
