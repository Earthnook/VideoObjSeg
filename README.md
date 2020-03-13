# Video Object Segmentation implementations and experiments

## Where this repo is from

This repo is forked from [STM](https://github.com/seoungwugoh/STM)

## Installation

**NOTE:** Currently, the settings are written for local device with cuda support, and there is a experiment tools written by me needs to be installed.

1. It is recommended using conda enviroment via 
    
    ```bash
    conda env create -f environment.yml
    ```

    If you want install [exptools](https://github.com/ziwenzhuang/exptools) manually, you may comment out the last line in `environment.yml` and follow the instructions of [exptools](https://github.com/ziwenzhuang/exptools).

2. Enter your conda environment via 

    ```bash
    conda activate vos
    ```

3. Install this repo as a library for all code in here 

    ```bash
    pip install -e .
    ```

## The design of the structure of this repository

1. This repo is focusing on experimenting the following problem

    - Given a `video` (RGB array with size: `t, C, W, H`) with all frames and maybe a `init_mask` (array with size: `t, W, H`),

        where the `init_mask` depends on the problem with index encoding
        
        and `c` is the channel of the video (usually 3).

        All pixel values are normalized to [0.0, 1.0] scale.

    - The model predict all `seg` (array with size: `t, n, W, H`) for the following frames of the `video`,

        where the `n` of the `seg` depends on the problem

2. Experiment paradigm

    a. Training



    b. Evaluation

3. Common definitions

    - **Image** An array with shape `C, H, W` where channel is at first

    - **Bounding Box** A 4-d array with following order: `(Hidx, Widx, Hlen, Wlen)`

        Then, cropping a images by index is like following
        ```Python
        image[:, Hidx:Hidx+Hlen, Widx:Widx+Wlen]
        ```
        **NOTE** This is the same as cropping array

## Usage

1. Show demo (evaluation) of [STM paper](https://arxiv.org/abs/1904.00607)

    a. Download weights

    ```bash
    wget -O STM_weights.pth "https://www.dropbox.com/s/mtfxdr93xc3q55i/STM_weights.pth?dl=1"
    ```

    b. Run demo for DAVIS-2017-trainval dataset

    ```bash
    python eval_DAVIS.py -g '1' -s val -y 17 -D [path/to/DAVIS]
    ```

2. Using Jupyter Notebook to debug using this conda environment

    ```bash
    conda activate vos
    python -m ipykernel install --user --name=vos
    ```
    