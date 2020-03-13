from vos.models import STM
from vos.models.correlation import conv2d_dw_group
from vos.utils.helpers import pad_divide_by

import torch
from torch import nn
from torch.nn import functional as F

class SiamQueryEncoder(STM.Encoder_Q):
    """ Siamese query encoder.
    Considering saving the memory, expand operation will be done here.
        And all output tensors will have leading dim (b*no, )
    """
    def __init__(self, train_bn= True):
        super(SiamQueryEncoder, self).__init__()

    def forward(self, in_f, target):
        """
        @ Args:
            frame: torch.Tensor with shape (b, C, H, W)
            target: torch.Tensor with shape (b, no, C, h, w) which is a cropped image
        @ Returns:
            feat: torch.Tensor with shape (b*no, c, h', w')
            other tensors are also with (b*no) as leading dimension
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
        b, no = target.shape[:2]
        target = target.view(b*no,
            target.shape[2],
            target.shape[3],
            target.shape[4]
        )
        f_tar = (target - self.mean) / self.std
        x_tar = self.conv1(f_tar) 
        x_tar = self.bn1(x_tar)
        c1_tar = self.relu(x_tar)   # 1/2, 64
        x_tar = self.maxpool(c1_tar)  # 1/4, 64
        r2_tar = self.res2(x_tar)   # 1/4, 256
        r3_tar = self.res3(r2_tar) # 1/8, 512
        r4_tar = self.res4(r3_tar) # 1/8, 1024

        # make output tensors with leading dim (b*no,)
        r4e = r4.unsqueeze(1).expand(b, no,-1,-1,-1)
        r3e = r3.unsqueeze(1).expand(b, no,-1,-1,-1)
        r2e = r2.unsqueeze(1).expand(b, no,-1,-1,-1)
        r4e = r4e.contiguous().view(b*no, r4e.shape[2],r4e.shape[3],r4e.shape[4])
        r3e = r3e.contiguous().view(b*no, r3e.shape[2],r3e.shape[3],r3e.shape[4])
        r2e = r2e.contiguous().view(b*no, r2e.shape[2],r2e.shape[3],r2e.shape[4])

        feat = conv2d_dw_group(r4e, r4_tar, pad= True) # same shape as r4e
        c_feat = torch.cat(
            (r4e, feat), dim= 1, # along channel dim.
        ) # C == 2048

        return c_feat, r3e, r2e

class conv2DBatchNormRelu(nn.Module):
    """ Code copied from
    https://github.com/meetshah1995/pytorch-semseg/blob/801fb200547caa5b0d91b8dde56b837da029f746/ptsemseg/models/utils.py#L87
    """
    def __init__(self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            is_batchnorm=True,
        ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class PyramidPooling(nn.Module):
    """ Given a tensor with shape (b, C, H, W), output tensor with shape (b, 2*C, H, W)
    refering to URL: https://arxiv.org/abs/1612.01105
    Code is highly based on 
        https://github.com/meetshah1995/pytorch-semseg/blob/801fb200547caa5b0d91b8dde56b837da029f746/ptsemseg/models/utils.py#L515
    """
    def __init__(self, 
            in_channels: int,
            pool_sizes, # A sequence of ints
            fusion_mode= "cat", 
            is_batchnorm= True,
        ):
        super(PyramidPooling, self).__init__()
        # processing scale from small to large

        bias = not is_batchnorm

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(
                conv2DBatchNormRelu(
                    in_channels= in_channels,
                    n_filters= int(in_channels / len(pool_sizes)),
                    k_size= 1,
                    stride= 1,
                    padding= 0,
                    bias=bias,
                    is_batchnorm=is_batchnorm,
                )
            )

        self.path_module_list = nn.ModuleList(self.paths)
        self.fusion_mode = fusion_mode
        self.pool_sizes = pool_sizes
    
    def forward(self, x):
        h, w = x.shape[2:]

        # let self be training mode in current implementation
        k_sizes = []
        strides = []
        for pool_size in self.pool_sizes:
            k_sizes.append((int(h / pool_size), int(w / pool_size)))
            strides.append((int(h / pool_size), int(w / pool_size)))

        # copied from 
        # https://github.com/meetshah1995/pytorch-semseg/blob/801fb200547caa5b0d91b8dde56b837da029f746/ptsemseg/models/utils.py#L515
        if self.fusion_mode == "cat":  # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != "icnet":
                    out = module(out)
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else:  # icnet: element-wise sum (including x)
            pp_sum = x

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                # out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != "icnet":
                    out = module(out)
                out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
                pp_sum = pp_sum + out

            return pp_sum

class Decoder(STM.Decoder):
    
    def __init__(self,
            in_channels,
            mdim,
        ):
        super(Decoder, self).__init__(2*in_channels, mdim)

        self.aspp = PyramidPooling(
            in_channels= 1024, # I infer from the code of STM
            pool_sizes= [6, 3, 2, 1],
            is_batchnorm= True,
        )


    def forward(self, r4, r3, r2):
        """ To meet the interface of STM, this network still accept more than 1 arguments,
        But they are not being used.
        @ Args:
            r4: the return of space-time memory read option.
        """
        r4pool = self.aspp(r4)
        return super(Decoder, self).forward(r4pool, r3, r2)


class EMN(STM.STM):
    """ Enhanced Memory Network： 
    https://youtube-vos.org/assets/challenge/2019/reports/YouTube-VOS-01_Enhanced_Memory_Network_for_Video_Segmentation.pdf
    """
    def __init__(self, **kwargs):
        super(EMN, self).__init__()
        self.Encoder_Q = SiamQueryEncoder()
        self.KV_Q_r4 = STM.KeyValue(2048, keydim= 128, valdim=512)
        self.Decoder = Decoder(1024, 256)

    def segment(self,
            frame,
            keys, values,
            num_objects,
            target
        ):
        """ NOTE: targets must has shape (B, K', C, H, W) where K' >= num_objects
        And target_bboxs has shape (B, K, 4) in (H, W, Hlen, Wlen) order
        """
        # if num_objects == 0, treat as 1 object
        num_objects = max(num_objects.max().item(), 1)
        B, K, keydim, T, H, W = keys.shape # B nope= 1
        B, K, valuedim, _, _, _ = values.shape
        # pad
        [frame], pad = pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4e, r3e, r2e = self.Encoder_Q(frame, target[:,:num_objects]) # B*no, dim, ...
        k4e, v4e = self.KV_Q_r4(r4e)   # B*no, dim, H/16, W/16

        # memory select kv: (B*no, C, T, H, W)
        m4, viz = self.Memory(
            keys[:,1:num_objects+1].contiguous().view(B*num_objects, keydim, T, H, W),
            values[:,1:num_objects+1].contiguous().view(B*num_objects, valuedim, T, H, W),
            k4e,
            v4e
        )
        logits = self.Decoder(m4, r3e, r2e)
        ps = F.softmax(logits, dim=1)[:,1] # B*no, h, w  
        #ps = indipendant possibility to belong to each object
        
        # Also move back to 4 dims
        logit = self.Soft_aggregation(ps, K, B) # B, K, H, W

        if pad[2]+pad[3] > 0:
            logit = logit[:,:,pad[2]:-pad[3],:]
        if pad[0]+pad[1] > 0:
            logit = logit[:,:,:,pad[0]:-pad[1]]

        return logit
