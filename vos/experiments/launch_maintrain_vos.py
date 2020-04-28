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
        n_gpu= 1,
        gpu_per_run= 1,
    )
    default_config = get_default_config()
    default_config["runner_kwargs"]["pretrain_optim_epochs"] = 0
    default_config["runner_kwargs"]["min_eval_itr"] = 1000
    

    # set up variants
    variant_levels = list()

    values = [
        ["EMN", True, True],
        # ["EMN", False, True],
        # ["EMN", True, False],
        # ["STM", False, False],
    ]
    dir_names = ["nn{}-atten{}-aspp{}".format(*v) for v in values]
    keys = [
        ("solution", ),
        ("model_kwargs", "use_target",),
        ("model_kwargs", "use_aspp",),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        [(480, 846) for _ in range(4)],
        # [(384, 384) for _ in range(4)],
    ]
    dir_names = ["img_res-{}".format(v[0]) for v in values]
    keys = [
        ("exp_image_size", ),
        ("videosynth_dataset_kwargs", "resolution", ),
        ("frame_skip_dataset_kwargs", "resolution", ),
        ("random_subset_kwargs", "resolution", ),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        # [True,  (24, 25), "interpolate"],
        # [False, (24, 25), "interpolate"],
        # [True,  ( 0, 25), "interpolate"],
        [False, ( 0, 25), "interpolate"],
    ]
    dir_names = ["data_spec-{}-{}-{}".format(*v) for v in values]
    keys = [
        ("frame_skip_dataset_kwargs", "update_on_full_view", ),
        ("frame_skip_dataset_kwargs", "skip_frame_range", ),
        ("videosynth_dataset_kwargs", "resize_method"),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        [1, 1, 1e-5, int(1e10), 0.9],
        # [4, 4, 1e-4, int(1e10), 0.9],
        # [24, 24, 1e-4, int(1e10), 0.9],
        # [20,20,5e-5, int(1e10), 0.9],
    ]
    dir_names = ["train_spec-{}-{}-{}-{}".format(*v[1:]) for v in values]
    keys = [
        ("pretrain_dataloader_kwargs", "batch_size"),
        ("dataloader_kwargs", "batch_size"),
        ("algo_kwargs", "learning_rate"),
        ("algo_kwargs", "lr_max_iter"),
        ("algo_kwargs", "lr_power"),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        [None],
        # ["/root/VideoObjSeg/data/weightfiles/STM_pretrain_51.82-52.93.pkl"],
        # ["/root/VideoObjSeg/data/weightfiles/STM_fulltrain_62.84-66.74.pkl"],
        # ["/root/VideoObjSeg/data/weightfiles/EMN_pretrain_54.50-59.29.pkl"],
        # ["/root/VideoObjSeg/data/weightfiles/EMN_fulltrain_60.54-64.98.pkl"],
    ]
    dir_names = [("preFalse" if i[0] is None else "preTrue") for i in values]
    keys = [
        ("pretrain_snapshot_filename", ),
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
