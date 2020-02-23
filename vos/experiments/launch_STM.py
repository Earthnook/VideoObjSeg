""" The entry point of configure and launch experiment.
"""
from exptools.launching.variant import VariantLevel, make_variants, update_config
from exptools.launching.affinity import encode_affinity
from exptools.launching.exp_launcher import run_experiments

from os import path

def get_default_config():
    root_path = "/p300/videoSeg_dataset/"
    dataset_root_path = path.join()"DAVIS-2017-trainval-480p/")
    return dict(
        pretrain_dataset_kwargs = dict(
            root= path.join(root_path, "COCO-2017-train/")
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
        algo_kwargs = dict(
            data_augment_kwargs= dict(
                angle_max= 90.,
                translate_max= 50.,
                scale_max= 2.,
                shear_max= 50.,
            ),
            n_frames= 3,
        ),
        runner_kwargs = dict(
            pretrain_optim_epochs= int(1e5),
            pretrain_dataloader_kwargs= dict(), # for torch DataLoader
            dataloader_kwargs= dict(), # for a customized DataLoader
            eval_dataloader_kwargs= dict(), # for a customized DataLoader
            eval_interval= 5,
            max_optim_epochs= int(1e5),
        )
    )

def main(args):
    experiment_title = "STM_reproduction"
    affinity_code = encode_affinity(
        n_cpu_core= 32,
        n_gpu= 8,
        gpu_per_run= 1,
        contexts_per_gpu= 4,
    )
    default_config = get_default_config()

    # set up variants
    variant_levels = list()

    values = [
        ["hopper",],
        ["pr2",],
        ["walker",],
    ]
    dir_names = ["{}".format(*v) for v in values]
    keys = [("env", "name")]
    variant_levels.append(VariantLevel(keys, values, dir_names))


    run_experiments(
        script="vos/experiments/STM.py",
        affinity_code=affinity_code,
        experiment_title=experiment_title+("--debug" if args.debug else ""),
        runs_per_setting=4,
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
        ptvsd.enable_attach(address=ip_address, redirect_output= True)
        # Pause the program until a remote debugger is attached
        ptvsd.wait_for_attach()
        print("Process attached, start running into experiment...", flush= True)
        ptvsd.break_into_debugger()

    main(args)
