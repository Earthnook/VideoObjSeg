from vos.utils.quick_args import save__init__args
from vos.runner.video_mask import VideoMaskRunner

from tqdm import tqdm
import torch
from torch.utils import data

class TwoStageRunner(VideoMaskRunner):
    """ The runner help with the STM training method.
    
    Details refering to https://arxiv.org/abs/1904.00607 and vos/algo/STM.py
    """
    def __init__(self,
            pretrain_optim_epochs,
            pretrain_dataloader,
            **kwargs
        ):
        save__init__args(locals())
        super(TwoStageRunner, self).__init__(**kwargs)

    def store_train_info(self, epoch_i, train_info, extra_info):
        super(TwoStageRunner, self).store_train_info(epoch_i, train_info, extra_info)
        self._store_extra_info(epoch_i, extra_info)

    def store_eval_info(self, epoch_i, eval_info, extra_info):
        super(TwoStageRunner, self).store_eval_info(epoch_i, eval_info, extra_info)
        self._store_extra_info(epoch_i, extra_info)

    def log_diagnostic(self, epoch_i):
        super(TwoStageRunner, self).log_diagnostic(epoch_i)
        self._log_extra_info(epoch_i)

    def _pre_train(self):
        for epoch_i in range(self.pretrain_optim_epochs):
            for batch_i, data in tqdm(enumerate(self.pretrain_dataloader)):
                train_info, extra_info = self.algo.pretrain(epoch_i, data)
                self.store_train_info(epoch_i, train_info, extra_info)
            self.log_diagnostic(epoch_i)

    def _main_train(self):
        for epoch_i in range(self.max_optim_epochs):
            for batch_i, data in tqdm(enumerate(self.dataloader)):
                train_info, extra_info = self.algo.train(epoch_i, data)
                self.store_train_info(epoch_i, train_info, extra_info)
            if not self.eval_dataset is None and epoch_i > 0 and epoch_i+1 % self.eval_interval:
                self.model.eval()
                for eval_data in tqdm(self.eval_dataloader):
                    eval_info, extra_info = self.algo.eval(epoch_i, eval_data)
                    self.store_eval_info(epoch_i, eval_info, extra_info)
                self.model.train()
            self.log_diagnostic(epoch_i)

    def train(self):
        """ one more image dataset to pre-train the network
        """
        self.startup()
        self._pre_train()
        torch.cuda.empty_cache()
        self._main_train()
        self.shutdown()
