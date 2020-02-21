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