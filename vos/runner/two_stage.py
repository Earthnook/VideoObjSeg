from vos.utils.helpers import load_pretrained_snapshot
from vos.utils.quick_args import save__init__args
from vos.runner.video_mask import VideoMaskRunner

from exptools.logging import logger

from tqdm import tqdm
import sys
import torch
from torch.utils import data

class TwoStageRunner(VideoMaskRunner):
    """ The runner help with the STM training method.
    
    Details refering to https://arxiv.org/abs/1904.00607 and vos/algo/STM.py
    """
    def __init__(self,
            pretrain_optim_epochs,
            pretrain_dataloader,
            max_predata_see= None,
            max_data_see= None, # if the dataset is too large, use these to limit the number of 
                                # iterations
            **kwargs
        ):
        save__init__args(locals())
        super(TwoStageRunner, self).__init__(**kwargs)

    def store_train_info(self, itr_i, train_info, extra_info):
        super(TwoStageRunner, self).store_train_info(itr_i, train_info, extra_info)
        self._store_extra_info(itr_i, extra_info, evaluate= False)

    def store_eval_info(self, itr_i, eval_info, extra_info):
        super(TwoStageRunner, self).store_eval_info(itr_i, eval_info, extra_info)
        self._store_extra_info(itr_i, extra_info, evaluate= True)

    def log_diagnostic(self, itr_i):
        super(TwoStageRunner, self).log_diagnostic(itr_i)
        self._log_extra_info(itr_i, evaluate= False)
        self._log_extra_info(itr_i, evaluate= True)

    def _train_loops(self,
            dataloader,
            eval_dataloader,
            max_optim_epochs,
            max_train_itr,
            itr_i,
        ):
        try:
            for epoch_i in range(max_optim_epochs):
                for batch_i, data in tqdm(enumerate(dataloader)):
                    itr_i += 1
                    train_info, extra_info = self.algo.train(itr_i, data)
                    self.store_train_info(itr_i, train_info, extra_info)

                    if not eval_dataloader is None and itr_i % self.eval_interval == 0 and itr_i > self.min_eval_itr:
                        self.model.eval()
                        for eval_data in tqdm(eval_dataloader):
                            # torch.cuda.empty_cache()
                            eval_info, extra_info = self.algo.eval(itr_i, eval_data)
                            self.store_eval_info(itr_i, eval_info, extra_info)
                            # torch.cuda.empty_cache()
                        self.model.train()
                        
                    if itr_i % self.log_interval == 0:
                        self.log_diagnostic(itr_i)
                    if max_train_itr is not None and itr_i >= max_train_itr:
                        return itr_i
                    if not sys.stdin.isatty() and sys.stdin.readline() == "next":
                        logger.log(f"User requested, move to next stage at iter: {itr_i}")
                        return itr_i
        except KeyboardInterrupt:
            logger.log(f"Keyboard interrupt at iter: {itr_i}, move to next stage")
        return itr_i

    def startup(self):
        if self.max_predata_see is not None \
            and self.max_predata_see < len(self.pretrain_dataloader) * self.pretrain_optim_epochs:
            self.max_pretrain_itr = self.max_predata_see // self.pretrain_dataloader.batch_size
        else:
            self.max_pretrain_itr = None
        if self.max_data_see is not None \
            and self.max_data_see < len(self.dataloader) * self.max_optim_epochs:
            self.max_train_itr = self.max_data_see // self.dataloader.batch_size
        else:
            self.max_train_itr = None
        super(TwoStageRunner, self).startup()

    def train(self, snapshot_filename= None):
        """ one more image dataset to pre-train the network
        """
        self.startup()
        
        if not snapshot_filename is None:
            try:
                itr_i = load_pretrained_snapshot(snapshot_filename, self.model, self.algo)
            except Exception as e:
                logger.log("load_snapshot_failed by \n {}".format(e))
                itr_i = 0
        else:
            itr_i = 0
        
        # pretrain
        itr_i = self._train_loops(
            dataloader= self.pretrain_dataloader,
            eval_dataloader= self.eval_dataloader,
            max_optim_epochs= self.pretrain_optim_epochs,
            max_train_itr= self.max_pretrain_itr,
            itr_i= itr_i,
        )
        logger.log("Finish pretraining, start main train at iteration: {}".format(itr_i))
        torch.cuda.empty_cache()
        # main train
        self._train_loops(
            dataloader= self.dataloader,
            eval_dataloader= self.eval_dataloader,
            max_optim_epochs= self.max_optim_epochs,
            max_train_itr= self.max_train_itr,
            itr_i= itr_i,
        )
        self.shutdown()
