from __future__ import division
#torch
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import os
import copy


def ToCuda(xs):
    # move the tensor to cuda
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda() 
    else:
        return xs


def pad_divide_by(in_list, d, in_size):
    out_list = []
    h, w = in_size
    if h % d > 0:
        new_h = h + d - h % d # = ceil(h/d) * d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d # = ceil(h/d) * d
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    for inp in in_list:
        out_list.append(F.pad(inp, pad_array))
    return out_list, pad_array



def overlay_davis(image,mask,colors=[255,0,0],cscale=2,alpha=0.4):
    """ Overlay segmentation on top of RGB image. from davis official"""
    # import skimage
    from scipy.ndimage.morphology import binary_erosion, binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours,:] = 0

    return im_overlay.astype(image.dtype)

def overlay_images(images, masks, alpha= 0.4):
    """ overlay a batch of images with some preset colors. Usually, the images pixel values are
    among [0, 1] scale.

    @ Args:
        images: np.ndarray with shape (b, C, H, W)
        masks: np.ndarray with shape (b, n, H, W) of index encoding

    @ returns:
        images: np.ndarray with shape (b, C, H, W)
    """
    n_foreground = masks.shape[1] - 1 # forget about background
    images = images.copy()

    # Sample from HSV color space and tranform value to RGB color space
    Hs = np.random.random(size= (n_foreground, 1))
    SVs = np.ones([n_foreground, 2])
    hsv = np.concatenate((Hs, SVs), axis= 1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)

    for mask_i in range(1, n_foreground+1):
        for channel_i in range(3):
            bin_mask = masks[:, mask_i]
            images[:, channel_i] += \
                (images[:, channel_i] - bin_mask*rgb[mask_i-1, channel_i]) * alpha

    return images

def load_snapshot(logdir, run_ID, model, algo):
    """ find proper snapshot file and load state dict to them
    NOTE: the file name is hard coded here, please make sure
    """
    files = [f for f in os.listdir(os.path.join(logdir, f"run_{run_ID}")) \
            if os.path.isfile(f) and ".pkl" in f]
    if len(files) < 1:
        return 0
    # Assuming there is only 1 .pkl file
    states = torch.load(os.path.join(logdir, files[0]))
    model.load_state_dict(states["model_state_dict"])
    algo.load_state_dict(states["algo_state_dict"])
    return states["itr_i"]