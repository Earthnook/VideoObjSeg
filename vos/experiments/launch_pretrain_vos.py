""" The entry point of configure and launch experiment.
"""
from exptools.launching.variant import VariantLevel, make_variants, update_config
from exptools.launching.affinity import encode_affinity, quick_affinity_code
from exptools.launching.exp_launcher import run_experiments

from os import path

def get_default_config():
    root_path = "/p300/videoObjSeg_dataset/"
    dataset_root_path = path.join(root_path, "DAVIS-2017-trainval-480p/")
    exp_image_size= (384, 384)
    return dict(
        exp_image_size= exp_image_size,
        solution= "STM",
        pretrain_snapshot_filename= None,
        coco_kwargs = dict(
            root= path.join(root_path, "COCO-2017-train/"),
            mode= "train",
            max_n_objects= 1,
            sort_anns= True,
        ),
        ecssd_kwargs = dict(
            root = path.join(root_path, "ECSSD/"),
        ),
        msra10k_kwargs = dict(
            root = path.join(root_path, "MSRA10K_Imgs_GT/"),
        ),
        train_dataset_kwargs = dict(
            root= dataset_root_path,
            mode= "train",
            max_n_objects= 1,
        ),
        eval_dataset_kwargs = dict(
            root= dataset_root_path,
            mode= "val",
            max_n_objects= 12,
        ),
        videosynth_dataset_kwargs = dict(
            n_frames= 3,
            resolution= exp_image_size,
            resize_method= "crop",
            affine_kwargs= dict(
                angle_max= 90.,
                translate_max= 50.,
                scale_max= 2.,
                shear_max= 50.,
            ),
            TPS_kwargs= dict(
                scale= 0.1,
                n_points= 5,
            ),
            dilate_scale= 5,
        ),
        frame_skip_dataset_kwargs = dict(
            n_frames= 3,
            skip_increase_interval= 50,
            max_clips_sample= 2,
            resolution= exp_image_size,
        ),
        random_subset_kwargs= dict(
            subset_len= 4,
            resolution= exp_image_size,
        ),
        pretrain_dataloader_kwargs= dict(
            batch_size= 4,
            shuffle= True,
            num_workers= 4,
        ), # for torch DataLoader
        dataloader_kwargs= dict(
            batch_size= 4,
            shuffle= True,
            num_workers= 4,
        ), # for a customized DataLoader
        eval_dataloader_kwargs= dict(
            batch_size= 1,
            num_workers= 4,
        ), # for a customized DataLoader
        model_kwargs= dict(
            train_bn= False,
        ),
        algo_kwargs= dict(
            include_bg_loss= False,
            clip_grad_norm= 1e9,
            learning_rate= 1e-5,
            weight_decay= 0,
            train_step_kwargs= dict(Mem_every= 1),
            eval_step_kwargs= dict(Mem_every= 5),
        ),
        runner_kwargs= dict(
            pretrain_optim_epochs= int(10),
            max_optim_epochs= int(20000),
            eval_interval= 20,
            log_interval= 5, # in terms of the # of calling algo.train()
            max_predata_see= None, # might make the training stop before reaching max_optim_epochs
            max_data_see= None,
        )
    )

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
        [0., 10., 0.05, 0., 0.1],
        # [5, 5, 0.05, 5, 0.1],
    ]
    dir_names = ["synth{}-{}-{}-{}-{}".format(*v) for v in values]
    keys = [
        ("videosynth_dataset_kwargs", "affine_kwargs", "angle_max"),
        ("videosynth_dataset_kwargs", "affine_kwargs", "translate_max"),
        ("videosynth_dataset_kwargs", "affine_kwargs", "scale_max"),
        ("videosynth_dataset_kwargs", "affine_kwargs", "shear_max"),
        ("videosynth_dataset_kwargs", "TPS_kwargs", "scale"),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        [(384, 384), ],
    ]
    dir_names = ["img_res-{},{}".format(*v[0]) for v in values]
    keys = [("exp_image_size",),]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        # ["EMN", ],
        ["STM", ],
    ]
    dir_names = ["NN{}".format(*v) for v in values]
    keys = [
        ("solution", ),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        [True, 1],
    ]
    dir_names = ["big_objects-{}{}".format(*v) for v in values]
    keys = [
        ("coco_kwargs", "sort_anns"),
        ("coco_kwargs", "max_n_objects"),
    ]
    variant_levels.append(VariantLevel(keys, values, dir_names))

    values = [
        # [1, 1],
        [4, 4],
    ]
    dir_names = ["b_size-{}".format(v[0]) for v in values]
    keys = [
        ("pretrain_dataloader_kwargs", "batch_size"),
        ("dataloader_kwargs", "batch_size"),
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
            variants[i]["train_dataset_kwargs"]["is_subset"] = True
            variants[i]["eval_dataset_kwargs"]["is_subset"] = True
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
