""" Some normalization function that shape arrays to make it possible to make batch.
"""
import numpy as np
import torch
from torch.nn import functional as F

def pad_bboxs(padding, bboxs: list):
    """ Given pad_up, pad_down, pad_left, pad_right, pad the bboxs
    """
    p_up, _, p_left, _ = padding

    p_bboxs = []
    for bbox in bboxs:
        if isinstance(bbox, torch.Tensor):
            p_bboxs.append(bbox + torch.tensor([p_up, p_left, 0, 0]))
        elif isinstance(bbox, np.ndarray):
            p_bboxs.append(bbox + np.array([p_up, p_left, 0, 0]))
        else:
            raise NotImplementedError
    if isinstance(bboxs, torch.Tensor):
        p_bboxs = torch.stack(p_bboxs)
    elif isinstance(bboxs, np.ndarray):
        p_bboxs = np.stack(p_bboxs)
    return p_bboxs

def crop_bboxs(cropping, bboxs: list):
    """
    @ Args:
        cropping: 4-elem tuple (c_up, c_left, c_Hlen, c_Wlen)
    """
    cH, cW, cHlen, cWlen = cropping
    
    c_bboxs = []
    for bbox in bboxs:
        h = max(0, bbox[0] - cH)
        w = max(0, bbox[1] - cW)
        hlen = min(bbox[2], cHlen - h)
        wlen = min(bbox[3], cWlen - w)
        c_bboxs.append(
            np.array(h, w, hlen, wlen) if isinstance(bbox, np.ndarray) \
            else torch.tensor(h, w, hlen, wlen)
        )
    if isinstance(bboxs, torch.Tensor):
        c_bboxs = torch.stack(c_bboxs)
    elif isinstance(bboxs, np.ndarray):
        c_bboxs = np.stack(c_bboxs)
    return c_bboxs

def ramdom_padding_CHW(output_size, images, bboxs= []):
    """ Randomly pad zeros at H and W dimensions depending on output_size
    @ Args:
        images: a sequencial object like tuple
            each element must have at least 2 dimensions, which means they are tensors
            with dim (..., H, W)
        bboxs: a sequence of 4-d vectors representing the bounding boxes of those images
            Since all images must have the same H and W, bboxs is not a nested sequence.
    @ Returns:
        0-th: a list of new images processed from your provided images
    """
    oH, oW = output_size
    H, W = images[0].shape[-2], images[0].shape[-1]
    shape_len = len(images[0].shape)
    assert oH > H or oW > W

    p_up = np.random.randint(oH - H + 1)
    p_down = oH - H - p_up
    p_left = np.random.randint(oW - W + 1)
    p_right = oW - W - p_left

    p_bboxs = pad_bboxs((p_up,p_down,p_left,p_right), bboxs= bboxs)
    if isinstance(images[0], torch.Tensor):
        pad = (p_left, p_right, p_up, p_down)
        p_images = [F.pad(image, tuple(pad)) for image in images]
    elif isinstance(images[0], np.ndarray):
        pad = [(0, 0)] * (shape_len - 2) + [(p_up, p_down), (p_left, p_right)]
        p_images = [np.pad(image, pad_width= pad) for image in images]
    else:
        raise NotImplementedError
    return p_images, p_bboxs

def ramdom_padding_HWC(output_size, images, bboxs= []):
    """ Randomly pad zeros at H and W dimensions depending on output_size
    @ Args:
        images: a sequencial object like tuple
            each element must have at least 2 dimensions, which means they are tensors
            with dims (H, W, c)
        bboxs: a sequence of 4-d vectors representing the bounding boxes of those images
            Since all images must have the same H and W, bboxs is not a nested sequence.
    @ Returns:
        0-th: a list of new images processed from your provided images
    """
    oH, oW = output_size
    H, W, _ = images[0].shape
    assert len(images[0].shape) == 3
    assert oH > H or oW > W

    p_up = np.random.randint(oH - H + 1)
    p_down = oH - H - p_up
    p_left = np.random.randint(oW - W + 1)
    p_right = oW - W - p_left

    p_bboxs = pad_bboxs((p_up,p_down,p_left,p_right), bboxs= bboxs)
    if isinstance(images[0], torch.Tensor):
        pad = (p_left, p_right, p_up, p_down)
        p_images = [F.pad(image, tuple(pad)) for image in images]
    elif isinstance(images[0], np.ndarray):
        pad = [(p_up, p_down), (p_left, p_right), (0, 0)]
        p_images = [np.pad(image, pad_width= pad) for image in images]
    else:
        raise NotImplementedError
    return p_images, p_bboxs

def random_crop(output_size, images, bboxs= []):
    """ Randomly crop images on H and W dimensions depends on output_size. Zero padings
    will be added randomly if the original size is not big enough.
    @ Args:
        images: a sequencial object like tuple
            each element must have at least 2 dimensions, which means they are tensors
            with dim (..., H, W)
        bboxs: a sequence of 4-d vectors representing the bounding boxes of those images
            Since all images must have the same H and W, bboxs is not a nested sequence.
    @ Returns:
        0-th: a list of new images processed from your provided images
    """
    oH, oW = output_size
    H, W = images[0].shape[-2], images[0].shape[-1]
    if oH > H and oW > W:
        images, bboxs = ramdom_padding_CHW(output_size, images= images, bboxs= bboxs)
        H, W = images[0].shape[-2], images[0].shape[-1]
    elif oH > H and oW <= W:
        images = ramdom_padding_CHW((oH, W), images= images, bboxs= bboxs)
        H, W = images[0].shape[-2], images[0].shape[-1]
    elif oH <= H and oW > W:
        images, bboxs = ramdom_padding_CHW((H, oW), images= images, bboxs= bboxs)
        H, W = images[0].shape[-2], images[0].shape[-1]

    H_start = np.random.randint(H-oH+1)
    W_start = np.random.randint(W-oW+1)

    c_images = [
        image[..., H_start:H_start+oH, W_start:W_start+oW] for image in images
    ]
    c_bboxs = crop_bboxs((H_start, W_start, oH, oW), bboxs= bboxs)
    return c_images, c_bboxs

random_crop_CHW = random_crop

def random_crop_HWC(output_size, images, bboxs= []):
    """ Randomly crop images on H and W dimensions depends on output_size. Zero padings
    will be added randomly if the original size is not big enough.
    @ Args:
        images: a sequencial object like tuple
            each element must have at least 2 dimensions, which means they are tensors
            with dim (H, W, c)
        bboxs: a sequence of 4-d vectors representing the bounding boxes of those images
            Since all images must have the same H and W, bboxs is not a nested sequence.
    @ Returns:
        0-th: a list of new images processed from your provided images
    """
    oH, oW = output_size
    H, W, _ = images[0].shape
    if oH > H and oW > W:
        images = ramdom_padding_HWC(output_size, images= images, bboxs= bboxs)
        H, W, _ = images[0].shape
    elif oH > H and oW <= W:
        images = ramdom_padding_HWC((oH, W), images= images, bboxs= bboxs)
        H, W, _ = images[0].shape
    elif oH <= H and oW > W:
        images = ramdom_padding_HWC((H, oW), images= images, bboxs= bboxs)
        H, W, _ = images[0].shape

    H_start = np.random.randint(H-oH+1)
    W_start = np.random.randint(W-oW+1)

    c_images = [
        image[H_start:H_start+oH, W_start:W_start+oW, :] for image in images
    ]
    c_bboxs = crop_bboxs((H_start, W_start, oH, oW), bboxs= bboxs)
    return c_images, c_bboxs
