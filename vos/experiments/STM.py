""" The script that runs the experiment
"""
import sys
from exptools.launching.affinity import affinity_from_code
from exptools.launching.variant import load_variant
from exptools.logging.context import logger_context

from vos.datasets.COCO import COCO
from vos.datasets.DAVIS import DAVIS_2017_TrainVal
from vos.datasets.frame_skip import FrameSkipDataset
from vos.models.STM import STM
from vos.algo.image_pretrain import ImagePretrainAlgo
from vos.runner.two_stage import TwoStageRunner

from torch.nn import DataParallel
from torch.utils.data import DataLoader

def build_and_train(affinity_code, log_dir, run_ID, **kwargs):
    affinity = affinity_from_code(affinity_code)
    config = load_variant(log_dir)

    # build the components for the experiment and run
    coco_train = COCO(**config["pretrain_dataset_kwargs"])
    davis_train = FrameSkipDataset(
        DAVIS_2017_TrainVal(**config["train_dataset_kwargs"]),
        **config["frame_skip_dataset_kwargs"]
    )
    davis_eval = DAVIS_2017_TrainVal(**config["eval_dataset_kwargs"])

    model = DataParallel(STM())

    algo = ImagePretrainAlgo(**config["algo_kwargs"])

    runner = TwoStageRunner(
        affinity= affinity,
        model= model,
        algo= algo,
        pretrain_dataloader= DataLoader(coco_train, **config["pretrain_dataloader_kwargs"]),
        dataloader= DataLoader(davis_train, 
            collate_fn= FrameSkipDataset.collate_fn,
            **config["dataloader_kwargs"]
        ),
        eval_dataloader= DataLoader(davis_eval, **config["eval_dataloader_kwargs"]),
        **config["runner_kwargs"]
    )

    name = "VOS_problem"
    with logger_context(log_dir, run_ID, name, log_params= config, snapshot_mode= "last"):
        runner.train()

def main(*args):
    build_and_train(*args)
if __name__ == "__main__":
    build_and_train(*sys.argv[1:])