""" Functionals to perform correlation
"""
from torch.nn import functional as F

def conv2d_dw_corr(x, temp):
    """ A batch-wise 2d image correlation in terms of vectors in channel dimension.
    @ Args:
        x: input tensor with shape (b, C, H, W)
        temp: template with shape (b, C, h, w)
    @ Returns:
        out: output feature with shape (b, C, outH, outW)
    """
    batch, channel = temp.shape[:2]
    x = x.view(1, batch*channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
    temp = temp.view(batch*channel, 1, temp.size(2), temp.size(3))  # (b*c) * 1 * H * W
    out = F.conv2d(x, temp, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))

    return out