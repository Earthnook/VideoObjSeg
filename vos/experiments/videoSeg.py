""" The script that runs the experiment
"""
import sys
import os
from exptools.launching.affinity import affinity_from_code, set_gpu_from_visibles, combine_affinity
from exptools.launching.variant import load_variant
from exptools.logging.context import logger_context

from vos.datasets.COCO import COCO
from vos.datasets.ECSSD import ECSSD
from vos.datasets.MSRA10K import MSRA10K
from vos.datasets.SBD import SBD
from vos.datasets.VOC import VOCSegmentation
from vos.datasets.DAVIS import DAVIS_2017_TrainVal
from vos.datasets.video_synth import VideoSynthDataset
from vos.datasets.frame_skip import FrameSkipDataset
from vos.datasets.random_subset import RandomSubset

from vos.models.loss import MultiObjectsBCELoss
from vos.models.STM import STM
from vos.algo.stm_train import STMAlgo
from vos.models.EMN import EMN
from vos.algo.emn_train import EMNAlgo

from vos.runner.two_stage import TwoStageRunner

from torch.nn import DataParallel
from torch.utils.data import DataLoader, ConcatDataset

def build_and_train(affinity_code, log_dir, run_ID, **kwargs):
    # Considering the batch size, You have to provide at least 4 gpus for
    # this experiment.
    affinity = affinity_from_code(affinity_code)
    if isinstance(affinity, list):
        affinity = combine_affinity(affinity)
    set_gpu_from_visibles(affinity.get("cuda_idx", 0))
    config = load_variant(log_dir)

    # build the components for the experiment and run
    if config["runner_kwargs"]["pretrain_optim_epochs"] > 0:
        datasets = list()
        config["coco_kwargs"].update({"max_n_objects": config["max_n_objects"]})
        config["sbd_kwargs"].update({"max_n_objects": config["max_n_objects"]})
        config["voc_kwargs"].update({"max_n_objects": config["max_n_objects"]})
        datasets.append(ECSSD(**config["ecssd_kwargs"]))
        datasets.append(SBD(**config["sbd_kwargs"]))
        datasets.append(MSRA10K(**config["msra10k_kwargs"]))
        datasets.append(VOCSegmentation(**config["voc_kwargs"]))
        if config["coco_kwargs"]["root"] is not None:
            datasets.append(COCO(**config["coco_kwargs"]))
        pretrain_dataset = ConcatDataset(datasets)
    else:
        pretrain_dataset = SBD(**config["sbd_kwargs"])

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
        algo = STMAlgo(
            loss_fn= MultiObjectsBCELoss(include_bg= config["algo_kwargs"]["include_bg_loss"]),
            **config["algo_kwargs"]
        )
    elif config["solution"] == "EMN":
        model = DataParallel(EMN(**config["model_kwargs"]))
        algo = EMNAlgo(
            loss_fn= MultiObjectsBCELoss(include_bg= config["algo_kwargs"]["include_bg_loss"]),
            **config["algo_kwargs"]
        )
    else:
        raise NotImplementedError("Cannnot deploy proper neural network solution")
    model.train()

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

    # load well-trained model if needed
    # import torch
    # model.load_state_dict(torch.load("/p300/VideoObjSeg_data/weightfiles/STM_weights.pth"), strict= False)
    # name = "Tune-targ{}-aspp{}-snap{}".format(
    
    name = "targ{}-aspp{}-snap{}".format(
        config["model_kwargs"]["use_target"],
        config["model_kwargs"]["use_aspp"],
        ("False" if config["pretrain_snapshot_filename"] is None else "True"),
    )
    with logger_context(log_dir, run_ID, name,
            log_params= config,
            snapshot_mode= "last",
        ):
        runner.train(snapshot_filename= config["pretrain_snapshot_filename"])
        # runner.train(snapshot_filename= os.path.join(log_dir, f"run_{run_ID}", "params.pkl"))


def main(*args):
    build_and_train(*args)
if __name__ == "__main__":
    build_and_train(*sys.argv[1:])