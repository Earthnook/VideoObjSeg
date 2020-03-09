from vos.algo.base import AlgoBase, TrainInfo, EvalInfo
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
            loss_fn= nn.BCELoss(),
            contour_weight= 1.0,
            train_step_kwargs= dict(),
            eval_step_kwargs= dict(),
            include_bg_loss= False,
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
        """ Given the statistics (in same shape (N, ...)) and caculate average value in this batch
            NOTE: the average is only taken on 0-th dimension
            Calculation refering to book: 
            Pattern Recognition and Computer Vision: Second Chinese Conference, PRCV page 423
        """
        N = pred.shape[0]
        gtruth = gtruth.astype(np.uint8).reshape(N, -1)
        pred = pred.astype(np.uint8).reshape(N, -1)
        # calculate region similarity (a.k.a Intersection over Unit)
        IoU = np.sum(gtruth & pred, axis= 1) / np.sum(gtruth | pred, axis= 1)

        # calculate contour accuracy
        intersect = np.sum(gtruth & pred, axis= 1)
        accuracy_rate = intersect / np.sum(pred, axis= 1)
        recall_rate = intersect / np.sum(gtruth, axis= 1)
        contour_acc = (np.reciprocal(self.contour_weight**2) + 1) * \
            (accuracy_rate * recall_rate) / (accuracy_rate + recall_rate)

        return dict(IoU= np.nanmean(IoU), contour_acc= np.nanmean(contour_acc))

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

        loss_idx = 0 if self.include_bg_loss else 1
        _, _, n, H, W = gtruths.shape
        p = preds[:,1:,loss_idx:].reshape(-1, H, W)
        g = gtruths[:,1:,loss_idx:].reshape(-1, H, W)
        performance_status = self.calc_performance(p, g)
    
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

        loss_idx = 0 if self.include_bg_loss else 1
        _, _, n, H, W = gtruths.shape
        p = preds[:,1:,loss_idx:].reshape(-1, H, W)
        g = gtruths[:,1:,loss_idx:].reshape(-1, H, W)
        performance_status = self.calc_performance(p, g)

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