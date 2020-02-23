import torch
from torch import nn
from torch.nn import functional as F

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
        return self.module(predicts, targets)