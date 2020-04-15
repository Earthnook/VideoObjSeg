from vos.utils.quick_args import save__init__args
from vos.utils.helpers import load_pretrained_snapshot

from exptools.logging import logger

import os
import psutil
from tqdm import tqdm
import torch
from torch.utils import data

class RunnerBase:
    """ The base runner for this supervised learning problem.
    """
    def __init__(self,
            affinity,
            model,
            algo,
            dataloader,
            eval_dataloader= None,
            log_interval= 50, # in terms of the # of calling algo.train()
            eval_interval= 10, # in terms of the # of calling algo.train()
            max_optim_epochs= 1e5,
                # The maximum number of training epochs.
                # NOTE: each train epoch will go through all data in dataset ONCE.
            
        ):
        save__init__args(locals())

    def startup(self):
        """ The procedure of initializing all components.
        And call to move to cuda if available
        """
        p = psutil.Process()
        
        # set experiment affinity
        try:
            if (self.affinity.get("master_cpus", None) is not None and
                    self.affinity.get("set_affinity", True)):
                p.cpu_affinity(self.affinity["master_cpus"])
            cpu_affin = p.cpu_affinity()
        except AttributeError:
            cpu_affin = "UNAVAILABLE affinity"
        logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: "
            f"{cpu_affin}.")
        # if affinity.get("master_torch_threads", None) is not None:
        #     torch.set_num_threads(affinity["master_torch_threads"])
        logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
            f"{torch.get_num_threads()}.")

        # configure cuda for this experiment
        logger.log(f"Runner {getattr(self, 'rank', '')} CUDA_VISIBLE_DEVICES: "
            f"{os.environ['CUDA_VISIBLE_DEVICES']}.")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.model.to(device= device)

        self.algo.initialize(self.model)

        self._train_infos = {k: list() for k in self.algo.train_info_fields}
        self._eval_infos = {k: list() for k in self.algo.eval_info_fields}

    def store_train_info(self, itr_i, train_info, extra_info):
        """ store train_info into attribute of self
        @ Args:
            train_info: a namedtuple
            extra_info: a dict
        """
        for k, v in self._train_infos.items():
            new_v = getattr(train_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])

    def store_eval_info(self, itr_i, eval_info, extra_info):
        """ store eval_info into attribute of self
        @ Args:
            eval_info: a namedtuple
            extra_info: a dict
        """
        for k, v in self._eval_infos.items():
            new_v = getattr(eval_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])

    def get_epoch_snapshot(self, itr_i):
        """ Collect all state needed for full checkpoint/snapshot of the training,
        including all model parameters and algorithm parameters
        """
        return dict(
            itr_i= itr_i,
            model_state_dict= self.model.state_dict(),
            algo_state_dict= self.algo.state_dict()
        )

    def save_epoch_snapshot(self, itr_i):
        """
        Calls the logger to save training checkpoint/snapshot (logger itself
        may or may not save, depending on mode selected).
        """
        logger.log("saving snapshot...")
        params = self.get_epoch_snapshot(itr_i)
        logger.save_itr_params(itr_i, params)
        logger.log("saved")

    def log_diagnostic(self, itr_i):
        """ write all informations into exact files using logging method
        """
        self.save_epoch_snapshot(itr_i)
        logger.record_tabular("Optim_itr", itr_i, itr_i)

        for k, v in self._train_infos.items():
            if not k.startswith("_"):
                logger.record_tabular_misc_stat(k, v, itr_i)
        self._train_infos = {k: list() for k in self._train_infos}
        
        for k, v in self._eval_infos.items():
            if not (k.startswith("_") or len(v) == 0):
                logger.record_tabular_misc_stat("Eval"+k, v, itr_i)
        self._eval_infos = {k: list() for k in self._eval_infos}

        logger.dump_tabular()

    def shutdown(self):
        """ Make sure all cleanup is done
        """
        pass

    def train(self, snapshot_filename= None):
        """ The main loop of the experiment.
        """
        self.startup()
        
        if not snapshot_filename is None:
            itr_i = load_pretrained_snapshot(snapshot_filename, self.model, self.algo)
        else:
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

        self.shutdown()