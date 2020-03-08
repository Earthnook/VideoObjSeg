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
    train_info_fields = tuple(f for f in TrainInfo._fields) # copy
    eval_info_fields = tuple(f for f in EvalInfo._fields) # copy

    def __init__(self,
            clip_grad_norm= 1e9,
            loss_fn= CrossEntropyOneHot(),
            contour_weight= 1.0,
            train_step_kwargs= dict(),
            eval_step_kwargs= dict(),
            **kwargs,
        ):
        save__init__args(locals())
        super(VideoObjSegAlgo, self).__init__(loss_fn= loss_fn, **kwargs)

    def step(self,
            frames: torch.Tensor,
            masks: torch.Tensor,
            n_objects: int,
            **kwargs,
        ):
        """ Different network has their different usage, so step should be different.
        And they should be able to tell you how to use the model an perform object tracking and
        segmentation task.
        """
        raise NotImplementedError

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
            **self.train_step_kwargs,
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
                masks= data["mask"].cpu().numpy(),
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
                **self.eval_step_kwargs,
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
                masks= data["mask"].cpu().numpy(),
                preds= preds,
                n_objects= data["n_objects"]
            )