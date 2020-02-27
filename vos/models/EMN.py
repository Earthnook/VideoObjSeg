from vos.models.STM import STM

import torch
from torch import nn
from torch.nn import functional as F

class SiamQueryEncoder(nn.Module):
    """ Siamese query encoder
    """
    def __init__(self):
        super(SiamQueryEncoder, self).__init__()

    def forward(self, in_f, init_object):
        """
        @ Args:
            frame: torch.Tensor with shape (b, C, H, W)
            init_object: torch.Tensor with shape (b, C, h, w) which is a cropped image
        """

        # calculating feature from query image
        f = (in_f - self.mean) / self.std
        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024

        # calculating feature from cropped target image
        f_tar = (init_object - self.mean) / self.std
        x_tar = self.conv1(f_tar) 
        x_tar = self.bn1(x_tar)
        c1_tar = self.relu(x_tar)   # 1/2, 64
        x_tar = self.maxpool(c1_tar)  # 1/4, 64
        r2_tar = self.res2(x_tar)   # 1/4, 256
        r3_tar = self.res3(r2_tar) # 1/8, 512
        r4_tar = self.res4(r3_tar) # 1/8, 1024

        feat = F.conv2d(r4, r4_tar)

        return feat


class PyramidDecoder(nn.Module):
    
    def __init__(self):
        super(PyramidDecoder, self).__init__()

    def forward(self, mem_out):
        """ To meet the interface of STM, this network still accept more than 1 arguments,
        But they are not being used.
        @ Args:
            mem_out: the return of space-time memory read option.
        """


class EMN(STM):
    """ Enhanced Memory Network： 
    https://youtube-vos.org/assets/challenge/2019/reports/YouTube-VOS-01_Enhanced_Memory_Network_for_Video_Segmentation.pdf
    """
    def __init__(self):
        super(EMN, self).__init__()

