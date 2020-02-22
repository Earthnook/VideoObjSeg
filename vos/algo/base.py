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
            learning_rate= 1e-5,
            weight_decay= 1e-2,
            loss_fn= nn.CrossEntropyLoss(),
        ):
        save__init__args(locals())

    def initialize(self,
            model: nn.Module,
        ):
        self.model = model
        self.optim = self.OptimCls(
            self._model.parameters(),
            lr= self.learning_rate,
            weight_decay= self.weight_decay
        )

    def state_dict(self):
        """ summarize current state for snapshot
        """
        return dict()

    def train(self, epoch_i, data):
        """ Perform one interation of optimization. Under most circumstance, it corresponding to
        one optim.step() call.

        @ Args:
            data: a dictionary with at least following k v pairs
                "frames": torch.Tensor with shape (b, t, C, H, W)
                "masks": torch.Tensor with shape (b, t, n, H, W), surving as ground truth in one-hot encoding

        @ returns:
            train_info: a namedtuple with numbered statistics
            extra_info: a dict depends on different problem, or algorithm
        """
        raise NotImplementedError

    def eval(self, epoch_i, data):
        """ Perform evaluation in terms of the given batch of data.

        @ Args:
            data: a dictionary with at least following k v pairs
                "frames": torch.Tensor with shape (b, t, C, H, W)
                "masks": torch.Tensor with shape (b, t, n, H, W), surving as ground truth in one-hot encoding

        @ returns:
            eval_info: a namedtuple
            extra_info: a dict depends on different problem, or algorithm
        """
        raise NotImplementedError