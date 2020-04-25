import os
import sys

import numpy as np
from davis2017.evaluation import DAVISEvaluation

############### all eval_DAVIS imports
import torch
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
import tqdm
import threading

### My libs
from vos.datasets.DAVIS import DAVIS_MO_Test
from vos.models.STM import STM
from vos.algo.stm_train import STMAlgo
from vos.models.EMN import EMN
from vos.algo.emn_train import EMNAlgo

GPU = "6"
outputroot = "/p300/VideoObjSeg_data/STM_test/pretrain_test/"

YEAR = "17"
SET = "val"

davisroot = "/p300/videoObjSeg_dataset/DAVIS-2017-trainval-480p"
dataset_eval = DAVISEvaluation(davis_root=davisroot, task="semi-supervised", gt_set="val", type="2017")

os.environ['CUDA_VISIBLE_DEVICES'] = GPU

torch.cuda.empty_cache()
palette = Image.open(davisroot + '/Annotations/480p/blackswan/00000.png').getpalette()

Testset = DAVIS_MO_Test(davisroot, resolution='480p', imset='20{}/{}.txt'.format(YEAR,SET), single_object=(YEAR==16))
Testloader = data.DataLoader(Testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

model = nn.DataParallel(STM())
algo = STMAlgo() # only use its step() method, so no need for any hyper-parameters

# model = nn.DataParallel(EMN())
# algo = EMNAlgo() # only use its step() method, so no need for any hyper-parameters

model.cuda()
model.eval() # turn-off BN
algo.initialize(model)

statedict_root = "/p300/VideoObjSeg_data/tmp_model/"
################################## 3 datasets stm pretrain
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200319/synth0.0-10.0-0.05-0.0-0.1/img_res-384,384/NNSTM/big_objects-True1/b_size-4/run_0"
################################## stm main only
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200326/img_res-384,384/NNSTM/b_size-4/pretrainFalse/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200403/img_res-384,384/NNSTM/b_size-4/pretrainFalse/run_0"
################################## 5 datasets stm pretrain
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200327/synth0.0-10.0-0.05-0.0-0.1/img_res-384,384/NNSTM/big_objects-True1/b_size-4/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200327/synth0.0-10.0-0.05-0.0-0.1/img_res-384,384/NNSTM/big_objects-True1/b_size-4/w_decay-0.0005/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200412/synth0.0-10.0-0.05-0.0-0.1/NNSTM/trainParam-24-1e-05-100000000000000000000-0.9/pixel_dilate-1/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200413/synth0.0-10.0-0.05-0.0-0.1/NNSTM/trainParam-4-1e-05-10000000000-0.9/pixel_dilate-1/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200413/synth0.0-10.0-0.05-0.0-0.1/NNSTM/trainParam-4-5e-05-10000000000-0.9/pixel_dilate-1/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200415/synth0.0-10.0-5.0-0.0-0.1/NNSTM/trainParam-24-1e-05-10000000000-0.9/pixel_dilate-5/run_0/20200415"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200415/synth3.0-5.0-5.0-0.0-0.1/NNSTM/trainParam-24-5e-05-10000000000-0.9/pixel_dilate-1/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200415/synth30.0-5.0-25.0-0.0-0.15/NNSTM/trainParam-24-1e-05-10000000000-0.9/pixel_dilate-5/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200416/synth3.0-5.0-5.0-0.0-0.1/NNSTM/trainParam-24-5e-05-10000000000-0.9/pixel_dilate-1/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200416/synth0.0-10.0-5.0-0.0-0.1/NNSTM/trainParam-24-1e-05-10000000000-0.9/pixel_dilate-5/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200416/synth1.0-10.0-5.0-1.0-0.1/NNSTM/trainParam-24-1e-05-10000000000-0.9/pixel_dilate-5/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200418/synth3.0-5.0-5.0-0.0-0.1/NNSTM/trainParam-24-5e-05-10000000000-0.9/pixel_dilate-1/run_0"
statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200425/synth0.0-10.0-5.0-0.0-0.1/NNSTM/trainParam-24-1e-05-10000000000-0.9/pixel_dilate-5/run_0"

################################## stm fulltrain
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200326/img_res-384,384/NNSTM/b_size-4/pretrainTrue/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200331/img_res-384,384/NNSTM/b_size-4/pretrainTrue/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200402/img_res-384,384/NNSTM/b_size-4/pretrainTrue/run_0"
#   statedict_root = "/root/VideoObjSeg/data/local/video_segmentation/20200422/NNSTM/train_spec-24-1e-05/pretrainTrue/run_0"

################################## 5 datasets emn pretrain
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200402/synth0.0-10.0-0.05-0.0-0.1/img_res-384,384/NNEMN/big_objects-True1/b_size-4/w_decay-0.0/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200407/synth0.0-10.0-0.05-0.0-0.1/img_res-384,384/NNEMN/big_objects-True1/b_size-4/w_decay-0.0/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200409/synth0.0-10.0-0.05-0.0-0.1/NNEMN/big_objects-True1/pixel_dilate-1/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200411/synth0.0-10.0-0.05-0.0-0.1/NNEMN/trainParam-24-5e-05-10000000000-0.9/pixel_dilate-1/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200420/synth3.0-5.0-5.0-0.0-0.1/NNEMN/trainParam-20-5e-05-10000000000-0.9/pixel_dilate-1/run_0"
# statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200425/synth0.0-10.0-5.0-0.0-0.1/NNEMN/trainParam-20-5e-05-10000000000-0.9/pixel_dilate-1/run_0"

################################## emn fulltrain
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200406/img_res-384,384/NNEMN/b_size-4/pretrainTrue/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200409/NNEMN/b_size-4/pretrainTrue/run_0"
#   statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200412/NNEMN/train_spec-20-5e-05/pretrainTrue/run_0"
# statedict_root = "/p300/VideoObjSeg_data/local/video_segmentation/20200425/NNEMN/train_spec-20-5e-05-10000000000-0.9/pretrainTrue/run_0"


to_save = [None, None, None] # a global object that enables multi-threading saving files
def save_result_to_output():
    seq_name, num_frames, pred = to_save[:]
    # save elements into outputroot sub-directories
    test_path = os.path.join(outputroot, seq_name)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    for f in range(num_frames):
        img_E = Image.fromarray(pred[f])
        img_E.putpalette(palette)
        img_E.save(os.path.join(test_path, '{:05d}.png'.format(f)))
    
def eval_davis(model_state_dict):
    torch.cuda.empty_cache()
    model.load_state_dict(model_state_dict)

    # construct fist thread, just for code consistency
    global to_save
    saving_thread = threading.Thread()
    saving_thread.start()
    
    # mantain the compute and saving schema
    for seq, V in enumerate(Testloader):
        torch.cuda.empty_cache()
        Fs, Ms, num_objects, info = V
        seq_name = info['name'][0]
        num_frames = info['num_frames'][0].item()
        print('video: {:2d} [{:15s}]: num_frames: {:3d}, num_objects: {:2d}'.format(seq, seq_name, num_frames, num_objects[0][0]),end= "\r")

        # compute
        with torch.no_grad():
            pred, _ = algo.step(
                frames= Fs,
                masks= Ms.type(torch.uint8),
                n_objects= num_objects,
                Mem_every=3, Mem_number=None
            )
        pred = np.argmax(pred[0].detach().cpu().numpy(), axis= 1).astype(np.uint8)
        
        # save
        saving_thread.join()
        del saving_thread
        to_save.pop(); to_save.pop(); to_save.pop()
        to_save.extend([seq_name, num_frames, pred])
        saving_thread = threading.Thread(target= save_result_to_output)
        saving_thread.start()
        
    saving_thread.join()
    del saving_thread
    print("Output to {} done".format(outputroot), flush= True)

    
J_thresh, F_thresh = 0.579, 0.621
# J_thresh, F_thresh = 0.692, 0.740
# J_thresh, F_thresh = 0.5182, 0.5293
# J_thresh, F_thresh = 0.63, 0.66
J_mean, F_mean = 0, 0
itr_is, Js, Fs = [], [], []
last_statedict = None
while True:
    # run to generate val output
    try:
        print("Loading weights", end= "\r")
        state_dict = torch.load(os.path.join(statedict_root, "params.pkl"))
#         state_dict = torch.load("/p300/VideoObjSeg_data/")
        print("Weight loaded at itr: {}".format(state_dict["itr_i"]))
        eval_davis(state_dict["model_state_dict"])
    except Exception as e:
        print(e, end= "\r")
        continue
    # run to evaluate the output
    metric_res = dataset_eval.evaluate(outputroot)
    J, F = metric_res['J'], metric_res['F']
    J_mean, F_mean = np.mean(J['M']), np.mean(F['M'])
    Js.append(J_mean); Fs.append(F_mean); itr_is.append(state_dict["itr_i"])
    
    last_statedict = state_dict
    print("J_mean: {:.4f}, F_mean: {:.4f}".format(J_mean, F_mean))
#     plt.plot(itr_is, Js, "r-")
#     plt.plot(itr_is, Fs, "b-")

    if (J_mean > J_thresh and F_mean > F_thresh):
        dst = os.path.join(statedict_root, "params-{:.2f}-{:.2f}.pkl".format(J_mean*100, F_mean*100))
        print("Save to: ", dst)
        torch.save(last_statedict, dst)