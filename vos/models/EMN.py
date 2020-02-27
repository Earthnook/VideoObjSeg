from vos.models.STM import STM
from vos.utils.helpers import pad_divide_by

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
        self.Encoder_Q = SiamQueryEncoder()
        self.Decoder = PyramidDecoder()

    def segment(self, frame, keys, values, num_objects): 
        # if num_objects == 0, treat as 1 object
        num_objects = max(num_objects.max().item(), 1)
        B, K, keydim, T, H, W = keys.shape # B nope= 1
        B, K, valuedim, _, _, _ = values.shape
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4 = self.Encoder_Q(frame)
        k4, v4 = self.KV_Q_r4(r4)   # B, dim, H/16, W/16

        # expand to ---  no, c, h, w
        k4e, v4e = k4.expand(B, num_objects,-1,-1,-1), v4.expand(B, num_objects,-1,-1,-1) 
        
        # make all objects into batch
        k4e, v4e = \
            k4e.view(B*num_objects, k4e.shape[2],k4e.shape[3],k4e.shape[4]), \
            v4e.view(B*num_objects, v4e.shape[2],v4e.shape[3],v4e.shape[4])

        # memory select kv: (B*no, C, T, H, W)
        m4, viz = self.Memory(
            keys[:,1:num_objects+1].view(B*num_objects, keydim, T, H, W),
            values[:,1:num_objects+1].view(B*num_objects, valuedim, T, H, W),
            k4e,
            v4e
        )
        logits = self.Decoder(m4)
        ps = F.softmax(logits, dim=1)[:,1] # B*no, h, w  
        #ps = indipendant possibility to belong to each object
        
        logit = self.Soft_aggregation(ps, K)[0] # B*K, H, W

        # recover from batch B, K, h_, w_
        logit = logit.view(B, K, logit.shape[1], logit.shape[2])

        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]

        return logit
