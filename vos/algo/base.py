from vos.utils.quick_args import save__init__args

import torch
from torch import nn
from torch import optim

from collections import namedtuple

TrainInfo = namedtuple("TrainInfo", ["loss", "gradNorm"])
EvalInfo = namedtuple("EvalInfo", ["loss"])

class AlgoBase:
    """ A basic implementation of the training algorithm
    """
    train_info_fields = tuple(f for f in TrainInfo._fields) # copy
    eval_info_fields = tuple(f for f in EvalInfo._fields) # copy

    def __init__(self,
            OptimCls= optim.Adam,
            learning_rate= 1e-3,
            weight_decay= 1e-2,
            loss_fn= nn.CrossEntropyLoss(),
        ):
        save__init__args(locals())

    def initialize(self,
            model: nn.Module,
        ):
        self._model = model
        self._optim = self.OptimCls(
            self._model.parameters(),
            lr= self.learning_rate,
            weight_decay= self.weight_decay
        )

    def train(self, data):
        """ Perform one interation of optimization. Under most circumstance, it corresponding to
        one optim.step() call.

        @ Args:
            data: a dictionary with at least following k v pairs
                "frames": torch.Tensor with shape (b, t, C, H, W)
                "masks": torch.Tensor with shape (b, t, n, H, W), surving as ground truth in one-hot encoding

        @ returns:
            opt_info: a namedtuple
        """
        raise NotImplementedError

    def eval(self, data):
        """ Perform evaluation in terms of the given batch of data.

        @ Args:
            data: a dictionary with at least following k v pairs
                "frames": torch.Tensor with shape (b, t, C, H, W)
                "masks": torch.Tensor with shape (b, t, n, H, W), surving as ground truth in one-hot encoding

        @ returns:
            eval_info: a namedtuple
        """
        raise NotImplementedError