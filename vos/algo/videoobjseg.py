from vos.algo.base import AlgoBase, TrainInfo, EvalInfo
from vos.models.loss import MultiObjectsBCELoss
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
            loss_fn= MultiObjectsBCELoss(include_bg= False),
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

    def calc_performance(self, pred, gtruth, n_objects, smooth= 1):
        """ Given the statistics and caculate average value in this batch
            NOTE: the average is only taken on 0-th dimension
            Calculation refering to book: 
            Pattern Recognition and Computer Vision: Second Chinese Conference, PRCV page 423
        @ Args:
            pred: ndarray with shape (B, T, N, H, W) with or without background channel
            grtuth: ndarray with the same shape as pred
            n_objects: ndarray with shape (B,) due to different n_objects in each data among batch
        """
        gtruth = gtruth.astype(np.bool)
        pred = pred.astype(np.bool)
        intersect = np.sum(gtruth & pred, axis= (-2, -1)) + smooth
        union = np.sum(gtruth | pred, axis= (-2, -1)) + smooth
        loss_idx = 0 if self.include_bg_loss else 1

        # calculate region similarity (a.k.a Intersection over Unit)
        IoU = intersect / union # (B, T, N)
        # calculate average IoU for each data in this batch
        IoU_mean = [] # a list of (T*n,)
        for b_i, iou in enumerate(IoU):
            iou_ = iou[:, loss_idx:(n_objects[b_i]+loss_idx)]
            IoU_mean.append(iou_.flatten())
        IoU_mean = np.nanmean(np.hstack(IoU_mean))

        # This modification referring to https://github.com/fperazzi/davis/blob/master/python/lib/davis/measures/jaccard.py#L29
        # And https://github.com/suhwan-cho/davis-evaluation/blob/master/davis2017/metrics.py#L37
        # NOTE: no need, smooth=1 make the same effect
        # IoU[np.isclose(union, 0)] = 1

        # calculate contour accuracy
        # NOTE: This calculation is not accurate due to experiment speed, 
        # please check official evaluation server
        accuracy_rate = intersect / (np.sum(pred, axis= (-2, -1)) + smooth)
        recall_rate = intersect / (np.sum(gtruth, axis= (-2, -1)) + smooth)
        contour_acc = 2 * (accuracy_rate * recall_rate) / (accuracy_rate + recall_rate) # (B, T, N)
        acc_mean = [] # a list of (T*n,)
        for b_i, acc in enumerate(contour_acc):
            acc_ = acc[:, loss_idx:(n_objects[b_i]+loss_idx)]
            acc_mean.append(acc_.flatten())
        acc_mean = np.nanmean(np.hstack(acc_mean))

        return dict(
            IoU= IoU_mean,
            contour_acc= acc_mean,
            IoU_each_frame= IoU, # add this term for debugging, will not effect logging mechanism
            contour_acc_frame= contour_acc,
        )

    def train(self, itr_i, data):
        """
        @ Args:
            data: a dictionary with following keys
                "video": a torch.Tensor with size (b, t, C, H, W), usually C == 3
                "mask": a torch.Tenwor with size (b, t, n, H, W) with one-hot encoding
                "n_objects": a batch-wise int telling how many objects in among batch of data
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

        # _, _, n, H, W = gtruths.shape
        # p = preds[:,1:,loss_idx:].reshape(-1, n-1, H, W)
        # g = gtruths[:,1:,loss_idx:].reshape(-1, n-1, H, W)
        performance_status = self.calc_performance(preds, gtruths, n_objects= data["n_objects"])
    
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

        # _, _, n, H, W = gtruths.shape
        # p = preds[:,1:,loss_idx:].reshape(-1, n-1, H, W)
        # g = gtruths[:,1:,loss_idx:].reshape(-1, n-1, H, W)
        performance_status = self.calc_performance(preds, gtruths, n_objects= data["n_objects"])

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