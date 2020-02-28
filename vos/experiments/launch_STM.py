""" The entry point of configure and launch experiment.
"""
from exptools.launching.variant import VariantLevel, make_variants, update_config
from exptools.launching.affinity import encode_affinity
from exptools.launching.exp_launcher import run_experiments

from os import path

def get_default_config():
    root_path = "/p300/videoObjSeg_dataset/"
    dataset_root_path = path.join(root_path, "DAVIS-2017-trainval-480p/")
    return dict(
        pretrain_dataset_kwargs = dict(
            root= path.join(root_path, "COCO-2017-train/"),
            mode= "train",
        ),
        train_dataset_kwargs = dict(
            root= dataset_root_path,
            mode= "train",
        ),
        eval_dataset_kwargs = dict(
            root= dataset_root_path,
            mode= "val"
        ),
        frame_skip_dataset_kwargs = dict(
            n_frames= 3,
            skip_increase_interval= 50,
            max_clips_sample= 2,
        ),
        random_subset_kwargs= dict(
            subset_len= 16,
        ),
        pretrain_dataloader_kwargs= dict(
            batch_size= 8,
            shuffle= True,
            num_workers= 8,
        ), # for torch DataLoader
        dataloader_kwargs= dict(
            batch_size= 4,
            shuffle= True,
            num_workers= 4,
        ), # for a customized DataLoader
        eval_dataloader_kwargs= dict(
            batch_size= 4,
            num_workers= 4,
        ), # for a customized DataLoader
        algo_kwargs = dict(
            data_augment_kwargs= dict(
                affine_kwargs= dict(
                    angle_max= 90.,
                    translate_max= 50.,
                    scale_max= 2.,
                    shear_max= 50.,
                ),
                n_frames= 2,
            ),
            clip_grad_norm= 1e9,
            learning_rate= 1e-5,
            weight_decay= 1e-3,
        ),
        runner_kwargs = dict(
            pretrain_optim_epochs= int(10),
            eval_interval= 10,
            log_interval= 20, # in terms of the # of calling algo.train()
            max_optim_epochs= int(20),
        )
    )

def main(args):
    experiment_title = "STM_reproduction"
    affinity_code = encode_affinity(
        n_cpu_core= 32,
        n_gpu= 8,
        gpu_per_run= 1,
        contexts_per_gpu= 1,
    )
    default_config = get_default_config()

    # set up variants
    variant_levels = list()

    # values = [
    #     ["hopper",],
    #     ["pr2",],
    #     ["walker",],
    # ]
    # dir_names = ["{}".format(*v) for v in values]
    # keys = [("env", "name")]
    # variant_levels.append(VariantLevel(keys, values, dir_names))

    variants, log_dirs = make_variants(*variant_levels)
    for i, variant in enumerate(variants):
        variants[i] = update_config(default_config, variant)
        if args.debug > 0:
            # make sure each complete iteration has gone through and easy for debug
            variants[i]["runner_kwargs"]["pretrain_optim_epochs"] = 5
            variants[i]["runner_kwargs"]["max_optim_epochs"] = 5
            variants[i]["runner_kwargs"]["eval_interval"] = 2
            variants[i]["runner_kwargs"]["log_interval"] = 4
            variants[i]["pretrain_dataset_kwargs"]["is_subset"] = True
            variants[i]["train_dataset_kwargs"]["is_subset"] = True
            variants[i]["eval_dataset_kwargs"]["is_subset"] = True
            variants[i]["pretrain_dataloader_kwargs"]["shuffle"] = False
            variants[i]["dataloader_kwargs"]["shuffle"] = False
            variants[i]["pretrain_dataloader_kwargs"]["num_workers"] = 0
            variants[i]["dataloader_kwargs"]["num_workers"] = 0
            variants[i]["eval_dataloader_kwargs"]["num_workers"] = 0
            
    run_experiments(
        script="vos/experiments/STM.py",
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
