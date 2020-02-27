""" Some normalization function that shape arrays to make it possible to make batch.
"""
import numpy as np
import torch
from torch.nn import functional as F

def ramdom_padding_CHW(output_size, *images):
    """ input images must have at least 2 dimensions, which means they are tensors with dim
    (..., H, W)
    """
    oH, oW = output_size
    H, W = images[0].shape[-2], images[0].shape[-1]
    shape_len = len(images[0].shape)
    assert oH > H or oW > W

    p_up = np.random.randint(oH - H)
    p_down = oH - H - p_up
    p_left = np.random.randint(oW - W)
    p_right = oW - W - p_left

    if isinstance(images[0], torch.Tensor):
        pad = [0, 0] * (shape_len - 2) + [p_up, p_down, p_left, p_right]
        return [
            F.pad(image, tuple(pad)) for image in images
        ]
    elif isinstance(images[0], np.ndarray):
        pad = [(0, 0)] * (shape_len - 2) + [(p_up, p_down), (p_left, p_right)]
        return [
            np.pad(image, pad_width= pad) for image in images
        ]
    else:
        raise NotImplementedError

def ramdom_padding_HWC(output_size, *images):
    """ input images must be with dims (H, W, c)
    """
    oH, oW = output_size
    H, W, _ = images[0].shape
    assert len(images[0].shape) == 3
    assert oH > H or oW > W

    p_up = np.random.randint(oH - H + 1)
    p_down = oH - H - p_up
    p_left = np.random.randint(oW - W + 1)
    p_right = oW - W - p_left

    if isinstance(images[0], torch.Tensor):
        pad = [p_up, p_down, p_left, p_right, 0, 0]
        return [
            F.pad(image, tuple(pad)) for image in images
        ]
    elif isinstance(images[0], np.ndarray):
        pad = [(p_up, p_down), (p_left, p_right), (0, 0)]
        return [
            np.pad(image, pad_width= pad) for image in images
        ]
    else:
        raise NotImplementedError

def random_crop(output_size, *images):
    """ input images must have at least 2 dimensions, which means they are tensors with dim
    (..., H, W)
    """
    oH, oW = output_size
    H, W = images[0].shape[-2], images[0].shape[-1]
    if oH > H and oW > W:
        images = ramdom_padding_CHW(output_size, *images)
        H, W = images[0].shape[-2], images[0].shape[-1]
    elif oH > H and oW <= W:
        images = ramdom_padding_CHW((oH, W), *images)
        H, W = images[0].shape[-2], images[0].shape[-1]
    elif oH <= H and oW > W:
        images = ramdom_padding_CHW((H, oW), *images)
        H, W = images[0].shape[-2], images[0].shape[-1]

    H_start = np.random.randint(H-oH+1)
    W_start = np.random.randint(W-oW+1)

    return [
        image[..., H_start:H_start+oH, W_start:W_start+oW] for image in images
    ]

random_crop_CHW = random_crop

def random_crop_HWC(output_size, *images):
    """ input images must be np.ndarray with dims (H, W, c)
    """
    oH, oW = output_size
    H, W, _ = images[0].shape
    if oH > H and oW > W:
        images = ramdom_padding_HWC(output_size, *images)
        H, W, _ = images[0].shape
    elif oH > H and oW <= W:
        images = ramdom_padding_HWC((oH, W), *images)
        H, W, _ = images[0].shape
    elif oH <= H and oW > W:
        images = ramdom_padding_HWC((H, oW), *images)
        H, W, _ = images[0].shape

    H_start = np.random.randint(H-oH+1)
    W_start = np.random.randint(W-oW+1)

    return [
        image[H_start:H_start+oH, W_start:W_start+oW, :] for image in images
    ]

def random_crop_256_CHW(*images):
    output_size = (256, 256)
    return random_crop_CHW(output_size, *images)

def random_crop_256_HWC(*images):
    output_size = (256, 256)
    return random_crop_HWC(output_size, *images)


    