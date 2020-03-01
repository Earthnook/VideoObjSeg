from vos.algo.base import AlgoBase, TrainInfo, EvalInfo
from vos.models.cross_entropy_onehot import CrossEntropyOneHot
from vos.utils.quick_args import save__init__args

from torchvision import transforms
from torchvision.transforms import functional as visionF
import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple
import numpy as np

TrainInfo = namedtuple("TrainInfo", ["loss", "gradNorm", "IoU", "contour_acc"])
EvalInfo = namedtuple("EvalInfo", ["loss", "IoU", "contour_acc"])

class VideoObjSegAlgo(AlgoBase):
    """ The algorithm re-producing STM training method
    https://arxiv.org/abs/1904.00607
    """
    train_info_fields = tuple(f for f in TrainInfo._fields) # copy
    eval_info_fields = tuple(f for f in EvalInfo._fields) # copy

    def __init__(self,
            clip_grad_norm= 1e9,
            loss_fn= CrossEntropyOneHot(),
            contour_weight= 1.0,
            **kwargs,
        ):
        save__init__args(locals())
        super(VideoObjSegAlgo, self).__init__(loss_fn= loss_fn, **kwargs)

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

        pred = (estimates.detach() == estimates.detach().max(dim= 1, keepdim= True)[0])
        pred = pred.to(dtype= torch.uint8)
        return pred, self.loss_fn(logits[:, 1:], masks[:, 1:]) # only query frames are used


    def calc_performance(self, pred, gtruth):
        """ Given the statistics, calculate the performace in terms of object segmentation.
        All inputs should be np.ndarray with the same shape and one-hot encoding.
            Calculation refering to book: 
            Pattern Recognition and Computer Vision: Second Chinese Conference, PRCV page 423
        """
        gtruth = gtruth.astype(np.uint8)
        pred = pred.astype(np.uint8)
        # calculate region similarity (a.k.a Intersection over Unit)
        IoU = np.sum(gtruth & pred) / np.sum(gtruth | pred)

        # calculate contour accuracy
        intersect = np.sum(pred & gtruth)
        accuracy_rate = intersect / np.sum(pred)
        recall_rate = intersect / np.sum(gtruth)
        contour_acc = (np.reciprocal(self.contour_weight**2) + 1) * \
            (accuracy_rate * recall_rate) / (accuracy_rate + recall_rate)

        return dict(IoU= IoU, contour_acc= contour_acc)

    def train(self, optim_i, data):
        """
        @ Args:
            data: a dictionary with following keys
                "video": a torch.Tensor with size (b, t, C, H, W), usually C == 3
                "mask": a torch.Tenwor with size (b, t, n, H, W) with one-hot encoding
                "n_objects": max number of objects
        """
        self.optim.zero_grad()
        preds, loss = self.step(
            frames= data["video"],
            masks= data["mask"],
            n_objects= data["n_objects"],
            Mem_every= 1,
        )
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optim.step()
        preds = preds.cpu().numpy()
        gtruths = data["mask"].cpu().numpy()

        performance_status = self.calc_performance(preds, gtruths)
    
        return TrainInfo(
                loss= loss.detach().cpu().numpy(), 
                gradNorm= grad_norm,
                IoU= performance_status["IoU"],
                contour_acc= performance_status["contour_acc"],
            ), \
            dict(
                videos= data["video"].cpu().numpy(),
                preds= preds,
                n_objects= data["n_objects"]
            )


    def eval(self, optim_i, data):
        """
        @ Args:
            data: a dictionary with following keys
                "video": a torch.Tensor with size (b, t, C, H, W), usually C == 3
                "mask": a torch.Tenwor with size (b, t, n, H, W) with one-hot encoding
                "n_objects": max number of objects
        """
        with torch.no_grad():
            preds, loss = self.step(
                frames= data["video"],
                masks= data["mask"],
                n_objects= data["n_objects"],
                Mem_every= 5,
            )
        preds = preds.cpu().numpy()
        gtruths = data["mask"].cpu().numpy()

        performance_status = self.calc_performance(preds, gtruths)

        return EvalInfo(
                loss= loss.cpu().numpy(),
                IoU= performance_status["IoU"],
                contour_acc= performance_status["contour_acc"],
            ), \
            dict(
                videos= data["video"].cpu().numpy(),
                preds= preds,
                n_objects= data["n_objects"]
            )