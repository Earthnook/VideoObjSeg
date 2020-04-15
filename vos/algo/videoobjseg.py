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

    def calc_performance(self, pred, gtruth, smooth= 0):
        """ Given the statistics (in same shape (N, ...)) and caculate average value in this batch
            NOTE: the average is only taken on 0-th dimension
            Calculation refering to book: 
            Pattern Recognition and Computer Vision: Second Chinese Conference, PRCV page 423
        """
        gtruth = gtruth.astype(np.bool)
        pred = pred.astype(np.bool)
        intersect = np.sum(gtruth & pred, axis= (-2, -1)) + smooth
        union = np.sum(gtruth | pred, axis= (-2, -1)) + smooth

        # calculate region similarity (a.k.a Intersection over Unit)
        IoU = intersect / union
        # This modification referring to https://github.com/fperazzi/davis/blob/master/python/lib/davis/measures/jaccard.py#L29
        # And https://github.com/suhwan-cho/davis-evaluation/blob/master/davis2017/metrics.py#L37
        IoU[np.isclose(union, 0)] = 1

        # calculate contour accuracy
        # NOTE: This calculation is not accurate due to experiment speed, 
        # please check official evaluation server
        accuracy_rate = intersect / (np.sum(pred, axis= (-2, -1)) + smooth)
        recall_rate = intersect / (np.sum(gtruth, axis= (-2, -1)) + smooth)
        contour_acc = 2 * (accuracy_rate * recall_rate) / (accuracy_rate + recall_rate)

        return dict(
            IoU= np.nanmean(IoU),
            contour_acc= np.nanmean(contour_acc),
            IoU_each_frame= IoU, # add this term for debugging, will not effect logging mechanism
            contour_acc_frame= contour_acc,
        )

    def train(self, itr_i, data):
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
        p = preds[:,1:,loss_idx:].reshape(-1, n-1, H, W)
        g = gtruths[:,1:,loss_idx:].reshape(-1, n-1, H, W)
        performance_status = self.calc_performance(p, g)
    
        return TrainInfo(
                loss= loss.detach().cpu().numpy(), 
                gradNorm= grad_norm,
                IoU= performance_status["IoU"],
                contour_acc= performance_status["contour_acc"],
            ), \
            dict(
                videos= data["video"].cpu().numpy()[:,1:],
                masks= data["mask"].cpu().numpy()[:,1:],
                preds= preds[:,1:],
                n_objects= data["n_objects"],
                IoU_each_frame= performance_status["IoU_each_frame"],
                contour_acc_frame= performance_status["contour_acc_frame"],
            )


    def eval(self, itr_i, data):
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
        p = preds[:,1:,loss_idx:].reshape(-1, n-1, H, W)
        g = gtruths[:,1:,loss_idx:].reshape(-1, n-1, H, W)
        performance_status = self.calc_performance(p, g)

        return EvalInfo(
                loss= loss.cpu().numpy(),
                IoU= performance_status["IoU"],
                contour_acc= performance_status["contour_acc"],
            ), \
            dict(
                videos= data["video"].cpu().numpy()[:,1:],
                masks= data["mask"].cpu().numpy()[:,1:],
                preds= preds[:,1:],
                n_objects= data["n_objects"],
                IoU_each_frame= performance_status["IoU_each_frame"],
                contour_acc_frame= performance_status["contour_acc_frame"],
            )