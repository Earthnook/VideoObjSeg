import torch
from torch import nn
from torch.nn import functional as F

# Deprecated, the same as BCELoss
class CrossEntropyOneHot(nn.Module):
    def __init__(self, *args, **kwargs):
        super(CrossEntropyOneHot, self).__init__()
        self.module = nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, predicts, ys: torch.Tensor):
        """
        @ Args:
            predicts: tensor with shape (b, T, n, H, W)
            ys: tensor with shape (b, T, n, H, W)
        """
        _, _, n, H, W = ys.shape
        ys = ys.reshape(-1, n, H, W)
        predicts = predicts.reshape(-1, n, H, W)

        _, targets = ys.max(dim= 1)
        return self.module(predicts, targets.to(dtype= torch.long))

class MultiObjectsBCELoss(nn.Module):
    """ In order to deal with the problem when num of objects are different among a
    batch, in VOS setting
    """
    def __init__(self, include_bg= False, smooth_factor= 1e-10, **kwargs):
        """
            include_bg: if True, background will be calculated into loss
            smooth_factor: in order to prevent 
        """
        super().__init__(**kwargs)
        self.include_bg = include_bg
        self.smooth_factor = smooth_factor
        self.bce_loss = nn.BCELoss()

    def forward(self, preds, gtruths, n_objects):
        """ This should be a tractible function that can be used to compute gradient in pytorch
        @ Args:
            preds, gtruths: one-hot mask as torch tensor with shape (B, T, N, H, W)
            n_objects: integer torch tensor with shape (B,)
        @ Returns:
            loss: a sclar as torch tensor
        """
        B, T, N, H, W = preds.shape
        assert n_objects.shape[0] == B, f"Wrong batch size for preds: {B} and n_objects: {n_objects.shape[0]}"
        loss_idx = 0 if self.include_bg else 1
        pred_batch = list() # compute loss only among (H, W) dimension
        gtruth_batch = list()
        for b_i in range(B):
            pred = preds[b_i, :, loss_idx:(n_objects[b_i]+loss_idx)] # (T, n_, H, W)
            gtruth = gtruths[b_i, :, loss_idx:(n_objects[b_i]+loss_idx)]
            pred_batch.append(pred.reshape(-1, H, W))
            gtruth_batch.append(gtruth.reshape(-1, H, W))

        loss = self.bce_loss(
            torch.cat(pred_batch, dim= 0) + self.smooth_factor,
            torch.cat(gtruth_batch, dim= 0)
        )
        return loss
