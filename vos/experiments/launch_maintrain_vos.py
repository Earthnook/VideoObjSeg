""" The entry point of configure and launch experiment.
"""
from exptools.launching.variant import VariantLevel, make_variants, update_config
from exptools.launching.affinity import encode_affinity, quick_affinity_code
from exptools.launching.exp_launcher import run_experiments

from os import path

from vos.experiments.launch_pretrain_vos import get_default_config

def main(args):
    experiment_title = "video_segmentation"
    affinity_code = encode_affinity(
        n_cpu_core= 48,
        n_gpu= 2,
        gpu_per_run= 2,
    )
    default_config = get_default_config()

    # set up variants
    variant_levels = list()

    values = [
        [(384, 384), ],
    ]
    dir_names = ["img_res-{},{}".format(*v[0]) for v in values]
    keys = [("exp_image_size",),]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        ["EMN", ],
        # ["STM", ],
    ]
    dir_names = ["NN{}".format(*v) for v in values]
    keys = [
        ("solution", ),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        # [1, 1, 1],
        [4, 4, 1],
        # [4, 4, 4],
    ]
    dir_names = ["b_size-{}".format(v[0]) for v in values]
    keys = [
        ("pretrain_dataloader_kwargs", "batch_size"),
        ("dataloader_kwargs", "batch_size"),
        ("train_dataset_kwargs", "max_n_objects"),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        # [None, 0],
        # ["/root/VideoObjSeg/data/weightfiles/STM_5ImgData_pretrain_62.1-65.6_DAVIS2017val.pkl", 0],
        ["/root/VideoObjSeg/data/weightfiles/EMN_5ImgData_pretrain_71.11-74.20_DAVIS2017val.pkl", 0],
    ]
    dir_names = [("pretrainFalse" if i[0] is None else "pretrainTrue") for i in values]
    keys = [
        ("pretrain_snapshot_filename", ),
        ("runner_kwargs", "pretrain_optim_epochs"),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    variants, log_dirs = make_variants(*variant_levels)
    for i, variant in enumerate(variants):
        variants[i] = update_config(default_config, variant)
        if args.debug > 0:
            # make sure each complete iteration has gone through and easy for debug
            variants[i]["runner_kwargs"]["pretrain_optim_epochs"] = 5
            variants[i]["runner_kwargs"]["max_optim_epochs"] = 5
            variants[i]["runner_kwargs"]["eval_interval"] = 2
            variants[i]["runner_kwargs"]["log_interval"] = 4
            variants[i]["pretrain_dataloader_kwargs"]["shuffle"] = False
            variants[i]["dataloader_kwargs"]["shuffle"] = False
            variants[i]["pretrain_dataloader_kwargs"]["num_workers"] = 0
            variants[i]["dataloader_kwargs"]["num_workers"] = 0
            variants[i]["eval_dataloader_kwargs"]["num_workers"] = 0
            variants[i]["random_subset_kwargs"]["subset_len"] = 2
            
    run_experiments(
        script="vos/experiments/videoSeg.py",
        affinity_code=affinity_code,
        experiment_title=experiment_title+("--debug" if args.debug else ""),
        runs_per_setting=1,
        variants=variants,
        log_dirs=log_dirs,
        debug_mode=args.debug,
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--debug', help= 'A common setting of whether to entering debug mode for remote attach',
        type= int, default= 0,
    )

    args = parser.parse_args()
    if args.debug > 0:
        # configuration for remote attach and debug
        import ptvsd
        import sys
        ip_address = ('0.0.0.0', 5050)
        print("Process: " + " ".join(sys.argv[:]))
        print("Is waiting for attach at address: %s:%d" % ip_address, flush= True)
        # Allow other computers to attach to ptvsd at this IP address and port.
        ptvsd.enable_attach(address=ip_address,)
        # Pause the program until a remote debugger is attached
        ptvsd.wait_for_attach()
        print("Process attached, start running into experiment...", flush= True)
        ptvsd.break_into_debugger()

    main(args)
