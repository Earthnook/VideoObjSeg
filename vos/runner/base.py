from vos.utils.quick_args import save__init__args

from exptools.logging import logger

import torch
from torch.utils import data

class RunnerBase:
    """ The base runner for this supervised learning problem.
    """
    def __init__(self,
            model,
            algo,
            DataLoaderCls= data.DataLoader,
            dataloader_kwargs= dict(),
            eval_dataloader_kwargs= dict(),
            eval_interval= 10,
                # The interval to store logs, params and do evaluation in terms of calling 
                # algo.step()
            max_optim_steps= 1e5,
                # The maximum number of training loops.
                # NOTE: each train loop will go through all data in dataset ONCE.
            
        ):
        save__init__args(locals())

    def startup(self, dataset, eval_dataset= None):
        """ The procedure of initializing all components.
        """
        self.algo.initialize(self.model)
        self.dataloader = self.DataLoaderCls(dataset, **self.dataloader_kwargs)
        if not eval_dataset is None:
            self.eval_dataloader = self.DataLoaderCls(eval_dataset, **self.eval_dataloader_kwargs)

        self._train_infos = {k: list() for k in self.algo.train_info_fields}
        self._eval_infos = {k: list() for k in self.algo.eval_info_fields}

    def store_train_info(self, epoch_i, train_info, extra_info):
        """ store train_info into attribute of self
        @ Args:
            train_info: a namedtuple
            extra_info: a dict
        """
        for k, v in self._train_infos.items():
            new_v = getattr(train_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])

    def store_eval_info(self, epoch_i, eval_info, extra_info):
        """ store eval_info into attribute of self
        @ Args:
            eval_info: a namedtuple
            extra_info: a dict
        """
        for k, v in self._eval_infos.items():
            new_v = getattr(eval_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])

    def get_epoch_snapshot(self, epoch_i):
        """ Collect all state needed for full checkpoint/snapshot of the training,
        including all model parameters and algorithm parameters
        """
        return dict(
            epoch_i= epoch_i,
            model_state_dict= self.model.state_dict(),
            algo_state_dict= self.algo.state_dict()
        )

    def save_epoch_snapshot(self, epoch_i):
        """
        Calls the logger to save training checkpoint/snapshot (logger itself
        may or may not save, depending on mode selected).
        """
        logger.log("saving snapshot...")
        params = self.get_epoch_snapshot(epoch_i)
        logger.save_itr_params(epoch_i, params)
        logger.log("saved")

    def log_diagnostic(self, epoch_i):
        """ write all informations into exact files using logging method
        """
        self.save_epoch_snapshot(epoch_i)
        logger.record_tabular("Epoch", epoch_i)

        for k, v in self._train_infos.items():
            if not k.startswith("_"):
                logger.record_tabular_misc_stat(k, v)
        self._train_infos = {k: list() for k in self._train_infos}
        
        for k, v in self._eval_infos.items():
            if not k.startswith("_"):
                logger.record_tabular_misc_stat("Eval"+k, v)
        self._eval_infos = {k: list() for k in self._eval_infos}

        logger.dump_tabular()

    def shutdown(self):
        """ Make sure all cleanup is done
        """
        pass

    def train(self, dataset, eval_dataset= None):
        """ The main loop of the experiment.
        """
        self.startup(dataset, eval_dataset)

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