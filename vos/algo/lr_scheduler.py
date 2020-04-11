""" Customized learning rate scheduler that are not implemented in pytorch """

import torch
from torch.optim.lr_scheduler import _LRScheduler

class PolyLR(_LRScheduler):
    """ Adjust learning rate by "poly" policy, refering to paper
    https://arxiv.org/abs/1506.04579
    and serch for "poly" in the full-text
    """
    def __init__(self, optimizer,
            init_lr= 1e-5, # The initial learning rate
            max_iter= 100, # The maximum number of calling step of this instance
            power= 0.9,
        ):
        self.init_lr = init_lr
        self.max_iter = max_iter
        self.power = power
        self.step_count = -1 # considering super class will call step() once
        super().__init__(optimizer)

    def step(self):
        self.step_count += 1
        assert self.step_count <= self.max_iter, "Call step() should not be more than {} times".format(self.max_iter)

        lr = self.init_lr * (1 - self.step_count / self.max_iter)**self.power

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr
