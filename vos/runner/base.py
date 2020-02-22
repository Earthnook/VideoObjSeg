from vos.utils.quick_args import save__init__args

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

    def startup(self, dataset= None, eval_dataset= None):
        """ The procedure of initializing all components.
        """
        self.algo.initialize(self.model)
        self.dataloader = self.DataLoaderCls(dataset, **self.dataloader_kwargs)
        if not eval_dataset is None:
            self.eval_dataloader = self.DataLoaderCls(eval_dataset, **self.eval_dataloader_kwargs)

    #     self.initialize_logging()

    # def initialize_logging(self):
    #     """ take a example of the optimization, and knows how to make logs
    #     """
    #     self._train_infos = {k: list() for k in self.algo.train_info_fields}
    #     self._eval_infos = {k: list() for k in self.algo.eval_info_fields}

    def log_train_info(self, opt_info):
        """ log opt_info into attribute of self
        @ Args:
            opt_info: a named tuple
        """
        raise NotImplementedError

    def log_eval_info(self, eval_info):
        """ log eval_info into attribute of self
        @ Args:
            eval_info: a named tuple
        """
        raise NotImplementedError

    def store_diagnostic(self):
        """ store all informations into exact files using logging method
        """
        raise NotImplementedError

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
                opt_info = self.algo.train(data)
                self.log_diagnostic(opt_info)
            if not eval_dataset is None and epoch_i > 0 and epoch_i % self.eval_interval:
                for eval_data in self.eval_dataloader:
                    eval_info = self.algo.eval(self, eval_data)
                    self.log_eval_info(eval_info)
            self.store_diagnostic()

        self.shutdown()