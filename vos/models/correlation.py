""" Functionals to perform correlation
"""
from torch.nn import functional as F

def conv2d_dw_group(x, temp, pad= False):
    """ A batch-wise 2d image correlation in terms of vectors in channel dimension.
    from https://github.com/foolwood/SiamMask/blob/0eaac33050fdcda81c9a25aa307fffa74c182e36/models/rpn.py#L32
    @ Args:
        x: input tensor with shape (b, C, H, W)
        temp: template with shape (b, C, h, w)
        pad: if pad, outH == H and outW == W
    @ Returns:
        out: output feature with shape (b, C, outH, outW)
    """
    b, C, H, W = x.shape
    x = x.view(1, b*C, x.size(2), x.size(3))  # 1, b*c, H, W
    temp = temp.view(b*C, 1, temp.size(2), temp.size(3))  # b*c, 1, h, w
    
    if pad:
        _, _, h, w = temp.shape
        out = F.conv2d(x, temp, groups=b*C, padding= (h//2, w//2))
        out = out[:,:,:H,:W] # 1, b*C, H, W
    else:
        out = F.conv2d(x, temp, groups=b*C) # 1, b*C, h', w'

    out = out.view(b, C, out.size(2), out.size(3))
    return out