from vos.utils.quick_args import save__init__args
from vos.runner.base import RunnerBase
from vos.runner.video_mask import VideoMaskRunner

class TwoStageRunner(RunnerBase, VideoMaskRunner):
    """ The runner help with the STM training method.
    
    Details refering to https://arxiv.org/abs/1904.00607 and vos/algo/STM.py
    """
    def __init__(self,
            DataLoaderCls,
            pretrain_optim_steps,
            pretrain_dataloader_kwargs, # with the same dataloader, maybe kwargs are different
            **kwargs
        ):
        save__init__args(locals())
        super(TwoStageRunner, self).__init__(**kwargs)

    def startup(self, pretrain_dataset, dataset, eval_dataset= None):
        super(TwoStageRunner, self).startup(dataset= dataset, eval_dataset= eval_dataset)
        self.pretrain_dataloader = self.DataLoaderCls(
            pretrain_dataset,
            **self.pretrain_dataloader_kwargs
        )

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
        for epoch_i in self.pretrain_optim_steps:
            for batch_i, data in enumerate(self.pretrain_dataloader):
                train_info, extra_info = self.algo.pretrain(epoch_i, data)
                self.store_train_info(epoch_i, train_info, extra_info)
            self.log_diagnostic(epoch_i)

        self.shutdown()

    def _main_train(self):
        for epoch_i in self.max_optim_steps:
            for batch_i, data in enumerate(self.dataloader):
                train_info, extra_info = self.algo.train(epoch_i, data)
                self.store_train_info(epoch_i, train_info, extra_info)
            if not self.eval_dataset is None and epoch_i > 0 and epoch_i+1 % self.eval_interval:
                for eval_data in self.eval_dataloader:
                    eval_info, extra_info = self.algo.eval(epoch_i, eval_data)
                    self.store_eval_info(epoch_i, eval_info, extra_info)
            self.log_diagnostic(epoch_i)

        self.shutdown()

    def train(self,
            pretrain_dataset,
            dataset,
            eval_dataset= None,
        ):
        """ one more image dataset to pre-train the network
        """
        self.startup(pretrain_dataset, dataset, eval_dataset)
        self._pre_train()
        self._main_train()
