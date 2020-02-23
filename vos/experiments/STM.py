""" The script that runs the experiment
"""
from exptools.launching.affinity import affinity_from_code
from exptools.launching.variant import load_variant
from exptools.logging.context import logger_context

from vos.datasets.COCO import COCO
from vos.datasets.DAVIS import DAVIS_2017_TrainVal
from vos.dataloader.frame_skip import FrameSkipDataLoader
from vos.models.STM import STM
from vos.algo.image_pretrain import ImagePretrainAlgo
from vos.runner.two_stage import TwoStageRunner

from torch.nn import DataParallel

def build_and_train(affinity_code, log_dir, run_ID, **kwargs):
    affinity = affinity_from_code(affinity_code)
    config = load_variant(log_dir)

    # build the components for the experiment and run
    coco_train = COCO(**config["pretrain_dataset_kwargs"])
    davis_train = DAVIS_2017_TrainVal(**config["train_dataset_kwargs"])
    davis_eval = DAVIS_2017_TrainVal(**config["eval_dataset_kwargs"])

    model = DataParallel(STM())

    algo = ImagePretrainAlgo(**config["algo_kwargs"])

    runner = TwoStageRunner(
        model= model,
        algo= algo,
        DataLoaderCls= FrameSkipDataLoader,
        **config["runner_kwargs"]
    )

    name = "VOS_problem"
    with logger_context(log_dir, run_ID, name, log_params= config, snapshot_mode= "last"):
        runner.train(
            pretrain_dataset= coco_train,
            dataset= davis_train,
            eval_dataset= davis_eval,
        )

def main(*args):
    build_and_train(*args)
if __name__ == "__main__":
    build_and_train(*sys.argv[1:])