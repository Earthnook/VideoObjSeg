from vos.utils.quick_args import save__init__args
from vos.runner.video_mask import VideoMaskRunner

from exptools.logging import logger

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

    def store_train_info(self, itr_i, train_info, extra_info):
        super(TwoStageRunner, self).store_train_info(itr_i, train_info, extra_info)
        self._store_extra_info(itr_i, extra_info)

    def store_eval_info(self, itr_i, eval_info, extra_info):
        super(TwoStageRunner, self).store_eval_info(itr_i, eval_info, extra_info)
        self._store_extra_info(itr_i, extra_info)

    def log_diagnostic(self, itr_i):
        super(TwoStageRunner, self).log_diagnostic(itr_i)
        self._log_extra_info(itr_i)

    def _pre_train(self):
        itr_i = 0
        for epoch_i in range(self.pretrain_optim_epochs):
            for batch_i, data in tqdm(enumerate(self.pretrain_dataloader)):
                itr_i += 1
                train_info, extra_info = self.algo.pretrain(itr_i, data)
                self.store_train_info(itr_i, train_info, extra_info)

                if not self.eval_dataloader is None and itr_i % self.eval_interval == 0:
                    self.model.eval()
                    for eval_data in tqdm(self.eval_dataloader):
                        eval_info, extra_info = self.algo.eval(itr_i, eval_data)
                        self.store_eval_info(itr_i, eval_info, extra_info)
                    self.model.train()
                
                if itr_i % self.log_interval == 0:
                    self.log_diagnostic(itr_i)

    def _main_train(self):
        itr_i = 0
        for epoch_i in range(self.max_optim_epochs):
            for batch_i, data in tqdm(enumerate(self.dataloader)):
                itr_i += 1
                train_info, extra_info = self.algo.train(itr_i, data)
                self.store_train_info(itr_i, train_info, extra_info)
                if not self.eval_dataloader is None and itr_i % self.eval_interval == 0:
                    self.model.eval()
                    for eval_data in tqdm(self.eval_dataloader):
                        eval_info, extra_info = self.algo.eval(itr_i, eval_data)
                        self.store_eval_info(itr_i, eval_info, extra_info)
                    self.model.train()
                    
                if itr_i % self.log_interval == 0:
                    self.log_diagnostic(itr_i)

    def train(self):
        """ one more image dataset to pre-train the network
        """
        self.startup()
        self._pre_train()
        logger.log("Finish pretraining, start main train")
        torch.cuda.empty_cache()
        self._main_train()
        self.shutdown()
