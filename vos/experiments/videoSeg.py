""" The script that runs the experiment
"""
import sys
import os
from exptools.launching.affinity import affinity_from_code, set_gpu_from_visibles
from exptools.launching.variant import load_variant
from exptools.logging.context import logger_context

from vos.datasets.COCO import COCO
from vos.datasets.ECSSD import ECSSD
from vos.datasets.MSRA10K import MSRA10K
from vos.datasets.multi_dataset import MultiDataset
from vos.datasets.DAVIS import DAVIS_2017_TrainVal
from vos.datasets.video_synth import VideoSynthDataset
from vos.datasets.frame_skip import FrameSkipDataset
from vos.datasets.random_subset import RandomSubset

from vos.models.STM import STM
from vos.algo.stm_train import STMAlgo
from vos.models.EMN import EMN
from vos.algo.emn_train import EMNAlgo

from vos.runner.two_stage import TwoStageRunner
from vos.utils.conbine_affinities import conbine_affinity
from vos.utils.helpers import load_snapshot

from torch.nn import DataParallel
from torch.utils.data import DataLoader

def build_and_train(affinity_code, log_dir, run_ID, **kwargs):
    # Considering the batch size, You have to provide at least 4 gpus for
    # this experiment.
    affinity = affinity_from_code(affinity_code)
    if isinstance(affinity, list):
        affinity = conbine_affinity(affinity)
    set_gpu_from_visibles(affinity.get("cuda_idx", 0))
    config = load_variant(log_dir)

    # build the components for the experiment and run
    coco_train = COCO(**config["coco_kwargs"]),
    ecssd = ECSSD(**config["ecssd_kwargs"])
    msra10k = MSRA10K(**config["msra10k_kwargs"])
    pretrain_dataset = MultiDataset(coco_train, ecssd, msra10k)

    config["videosynth_dataset_kwargs"].update({"resolution": config["exp_image_size"]})
    train_dataset = VideoSynthDataset(
        pretrain_dataset,
        **config["videosynth_dataset_kwargs"],
    )
    config["frame_skip_dataset_kwargs"].update({"resolution": config["exp_image_size"]})
    davis_train = FrameSkipDataset(
        DAVIS_2017_TrainVal(
            **config["train_dataset_kwargs"],
        ),
        **config["frame_skip_dataset_kwargs"]
    )
    config["random_subset_kwargs"].update({"resolution": config["exp_image_size"]})
    davis_eval = RandomSubset(
        DAVIS_2017_TrainVal(**config["eval_dataset_kwargs"]),
        **config["random_subset_kwargs"],
    )

    if config["solution"] == "STM":
        model = DataParallel(STM(**config["model_kwargs"]))
        algo = STMAlgo(**config["algo_kwargs"])
    elif config["solution"] == "EMN":
        model = DataParallel(EMN(**config["model_kwargs"]))
        algo = EMNAlgo(**config["algo_kwargs"])
    else:
        raise NotImplementedError("Cannnot deploy proper neural network solution")
    model.train()

    # load parameters if available
    # itr_i = load_snapshot(log_dir, run_ID, model, algo)
    itr_i = 0

    runner = TwoStageRunner(
        affinity= affinity,
        model= model,
        algo= algo,
        pretrain_dataloader= DataLoader(train_dataset, **config["pretrain_dataloader_kwargs"]),
        dataloader= DataLoader(davis_train, 
            **config["dataloader_kwargs"]
        ),
        eval_dataloader= DataLoader(davis_eval,
            **config["eval_dataloader_kwargs"]
        ),
        **config["runner_kwargs"]
    )

    name = "Exp-{}".format(config["solution"])
    with logger_context(log_dir, run_ID, name,
            log_params= config,
            snapshot_mode= "last",
            itr_i= itr_i
        ):
        runner.train(itr_i)


def main(*args):
    build_and_train(*args)
if __name__ == "__main__":
    build_and_train(*sys.argv[1:])