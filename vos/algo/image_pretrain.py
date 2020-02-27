from vos.algo.base import AlgoBase, TrainInfo, EvalInfo
from vos.models.cross_entropy_onehot import CrossEntropyOneHot
from vos.utils.quick_args import save__init__args

from torchvision import transforms
from torchvision.transforms import functional as visionF
import torch
from torch import nn
from torch.nn import functional as F
from collections import namedtuple
import numpy as np

TrainInfo = namedtuple("TrainInfo", ["loss", "gradNorm", "IoU", "contour_acc"])
EvalInfo = namedtuple("EvalInfo", ["loss", "IoU", "contour_acc"])

class ImagePretrainAlgo(AlgoBase):
    """ The algorithm re-producing STM training method
    https://arxiv.org/abs/1904.00607
    """
    train_info_fields = tuple(f for f in TrainInfo._fields) # copy
    eval_info_fields = tuple(f for f in EvalInfo._fields) # copy

    def __init__(self,
            data_augment_kwargs= dict(
                affine_kwargs = dict(
                    angle_max= 180.,
                    translate_max= 50.,
                    scale_max= 2., # NOTE: this is the exponent of e
                    shear_max= 50.
                ), # a dict of kwargs providing for torchvision.transforms.functional.affine
                n_frames= 2, # 2 as minimum, usually 3
            ),
            clip_grad_norm= 1e9,
            loss_fn= CrossEntropyOneHot(),
            contour_weight= 1.0,
            **kwargs,
        ):
        save__init__args(locals())
        super(ImagePretrainAlgo, self).__init__(loss_fn= loss_fn, **kwargs)

        self.to_pil_image = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def step(self,
            frames: torch.Tensor,
            masks: torch.Tensor,
            n_objects: int,
            Mem_every=None, Mem_number=None,
        ):
        """ go through the data and calculate the loss (torch.Variable)
        And all arrays in this function are torch.Tensors
        @ Args:
            frames: batch-wise, shape (b, t, C, H, W)
            masks: batch-wise, shape (b, t, n, H, W)
            n_objects: A int telling how many objects (max) in this batch of data
        """
        n_frames = frames.size()[1]
        # initialize storage tensors
        if Mem_every:
            to_memorize = [int(i) for i in np.arange(0, n_frames, step=Mem_every)]
        elif Mem_number:
            to_memorize = [int(round(i)) for i in np.linspace(0, n_frames, num=Mem_number+2)[:-1]]
        else:
            raise NotImplementedError

        estimates = torch.zeros_like(masks, dtype= torch.float32)
        logits = torch.zeros_like(masks, dtype= torch.float32)
        estimates[:, 0] = masks[:, 0]

        for t in range(1, n_frames):
            # memorize
            prev_key, prev_value = self.model(
                frames[:, t-1],
                estimates[:, t-1],
                n_objects,
            )

            if t-1 == 0:
                this_keys, this_values = prev_key, prev_value
            else:
                this_keys = torch.cat([keys, prev_key], dim= 3)
                this_values = torch.cat([values, prev_value], dim= 3)

            # segment
            logit = self.model(
                frames[:, t],
                this_keys,
                this_values,
                n_objects,
            )
            logits[:, t] = logit
            estimates[:, t] = F.softmax(logit, dim= 1)

            # update memory
            if t-1 in to_memorize:
                keys, values = this_keys, this_values

        pred = (estimates.detach() == estimates.detach().max(dim= 1, keepdim= True)[0])
        pred = pred.to(dtype= torch.uint8)
        return pred, self.loss_fn(logits[:, 1:], masks[:, 1:]) # only query frames are used

    def random_transforms(self, image, mask):
        """ randomly generate a transform arguments, and apply it to both image and mask.
        Considering torchvision transform can only transform a single image, in requires
        some meanuvers to do with mask.
            NOTE: both arguments are torh.Tensor
        """
        affine_ranges = self.data_augment_kwargs["affine_kwargs"]
        # NOTE: np.random.uniform generates value for this dictionary
        affine_kwargs = dict(
            angle = np.random.uniform(
                low= -affine_ranges["angle_max"],
                high= affine_ranges["angle_max"],
                size= (1,)
            ).item(),
            translate = np.random.uniform(
                low= -affine_ranges["translate_max"],
                high= affine_ranges["translate_max"],
                size= (2,)
            ).tolist(),
            scale = np.random.uniform(
                low= np.exp(-affine_ranges["scale_max"]),
                high= np.exp(affine_ranges["scale_max"]),
                size= (1,)
            ).item(),
            shear = np.random.uniform(
                low= -affine_ranges["shear_max"],
                high= affine_ranges["shear_max"],
                size= (1,)
            ).item(),
        )

        image = self.to_tensor(visionF.affine(self.to_pil_image(image), **affine_kwargs))

        n, H, W = mask.shape
        layers_of_mask = []
        for m in mask:
            jittered_m = visionF.affine(self.to_pil_image(m), **affine_kwargs)
            layers_of_mask.extend([self.to_tensor(jittered_m)])
        mask = torch.cat(layers_of_mask, 0).to(dtype= torch.uint8)

        return image, mask

    def synth_videos(self, images, masks):
        """ Synthesize video clips by torch images. Return a torch.Tensor as a batch of
        video clips
        """
        # pil_images = [self.to_pil_image(img) for img in images]
        # pil_masks = [self.to_pil_image(msk) for msk in masks]
        videos, m_videos = [], []
        with torch.no_grad():
            for image, mask in zip(images, masks):
                video, m_video = [image], [mask]
                for frame_i in range(self.data_augment_kwargs["n_frames"]):
                    frame, m_frame = self.random_transforms(image, mask)
                    video.append(frame)
                    m_video.append(m_frame)
                videos.append(torch.stack(video))
                m_videos.append(torch.stack(m_video))
        videos = torch.stack(videos)
        m_videos = torch.stack(m_videos)
        # the returned videos should be batch-wise
        return videos, m_videos

    def calc_performance(self, pred, gtruth):
        """ Given the statistics, calculate the performace in terms of object segmentation.
        All inputs should be np.ndarray with the same shape and one-hot encoding.
            Calculation refering to book: 
            Pattern Recognition and Computer Vision: Second Chinese Conference, PRCV page 423
        """
        # calculate region similarity (a.k.a Intersection over Unit)
        IoU = np.sum(gtruth & pred) / np.sum(gtruth | pred)

        # calculate contour accuracy
        intersect = np.sum(pred & gtruth)
        accuracy_rate = intersect / np.sum(pred)
        recall_rate = intersect / np.sum(gtruth)
        contour_acc = (np.reciprocal(self.contour_weight**2) + 1) * \
            (accuracy_rate * recall_rate) / (accuracy_rate + recall_rate)

        return dict(IoU= IoU, contour_acc= contour_acc)
        
    def pretrain(self, optim_i, data):
        """ As the paper described, pretrain on images is the first stage.
        @ Args:
            data: a dictionary with following keys
                "image": a torch.Tensor with size (b, C, H, W), usually C==3
                "mask": a torch.Tenwor with size (b, n, H, W) with one-hot encoding
                "n_objects": max number of objects
        """
        videos, masks = self.synth_videos(data["image"], data["mask"])

        self.optim.zero_grad()
        preds, loss = self.step(
            frames= videos,
            masks= masks,
            n_objects= data["n_objects"],
            Mem_every= 1,
        )
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optim.step()
        preds = preds.cpu().numpy()
        gtruths = masks.cpu().numpy()

        performance_status = self.calc_performance(preds, gtruths)

        return TrainInfo(
                loss= loss.detach().cpu().numpy(), 
                gradNorm= grad_norm,
                IoU= performance_status["IoU"],
                contour_acc= performance_status["contour_acc"],
            ), \
            dict(
                videos= videos.cpu().numpy(),
                preds= preds,
                n_objects= data["n_objects"]
            )

    def train(self, optim_i, data):
        """
        @ Args:
            data: a dictionary with following keys
                "video": a torch.Tensor with size (b, t, C, H, W), usually C == 3
                "mask": a torch.Tenwor with size (b, t, n, H, W) with one-hot encoding
                "n_objects": max number of objects
        """
        self.optim.zero_grad()
        preds, loss = self.step(
            frames= data["video"],
            masks= data["mask"],
            n_objects= data["n_objects"],
            Mem_every= 1,
        )
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optim.step()
        preds = preds.cpu().numpy()
        gtruths = data["mask"]

        performance_status = self.calc_performance(preds, gtruths)
    
        return TrainInfo(
                loss= loss.detach().cpu().numpy(), 
                gradNorm= grad_norm,
                IoU= performance_status["IoU"],
                contour_acc= performance_status["contour_acc"],
            ), \
            dict(
                videos= data["video"].cpu().numpy(),
                preds= preds,
                n_objects= data["n_objects"]
            )


    def eval(self, optim_i, data):
        """
        @ Args:
            data: a dictionary with following keys
                "video": a torch.Tensor with size (b, t, C, H, W), usually C == 3
                "mask": a torch.Tenwor with size (b, t, n, H, W) with one-hot encoding
                "n_objects": max number of objects
        """
        with torch.no_grad():
            preds, loss = self.step(
                frames= data["video"],
                masks= data["mask"],
                n_objects= data["n_objects"],
                Mem_every= 5,
            )
        preds = preds.cpu().numpy()
        gtruths = data["mask"]

        performance_status = self.calc_performance(preds, gtruths)

        return EvalInfo(
                loss= loss.cpu().numpy(),
                IoU= performance_status["IoU"],
                contour_acc= performance_status["contour_acc"],
            ), \
            dict(
                videos= data["video"].cpu().numpy(),
                preds= preds,
                n_objects= data["n_objects"]
            )