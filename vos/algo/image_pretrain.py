from vos.algo.base import AlgoBase, TrainInfo, EvalInfo
from vos.models.cross_entropy_onehot import CrossEntropyOneHot
from vos.utils.quick_args import save__init__args

from torchvision import transforms
from torchvision.transforms import functional as visionF
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class ImagePretrainAlgo(AlgoBase):
    """ The algorithm re-producing STM training method
    https://arxiv.org/abs/1904.00607
    """
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

        estimates = torch.zeros_like(masks)
        estimates[:, 0] = masks[:, 0]

        for t in range(1, n_frames):
            # memorize
            prev_key, prev_value = self.model(
                frames[:, t-1],
                estimates[:, t-1],
                torch.tensor([n_objects])
            )

            if t == 0:
                this_keys, this_values = prev_key, prev_value
            else:
                this_keys = torch.cat([keys, prev_key], dim= 3)
                this_values = torch.vat([values, prev_value], dim= 3)

            # segment
            logit = self.model(
                frames[:, t],
                this_keys,
                this_values,
                torch.tensor([n_objects]),
            )
            estimates[:, 0] = F.softmax(logit, dim= 1)

            # update memory
            if t-1 in to_memorize:
                keys, values = this_keys, this_values

        pred = np.argmax(estimates.detach().cpu().numpy(), axis= 1).astype(np.uint8)
        return pred, self.loss_fn(estimates, masks)

    def random_transforms(self, image, mask):
        """ randomly generate a transform arguments, and apply it to both image and mask
        """
        affine_ranges = self.data_augment_kwargs["affine_kwargs"]
        # NOTE: np.random.uniform generates value for this dictionary
        affine_kwargs = dict(
            angle = np.random.uniform(
                low= -affine_ranges["angle_max"],
                high= affine_ranges["angle_max"],
                size= (1,)
            ),
            translate = np.random.uniform(
                low= -affine_ranges["translate_max"],
                high= affine_ranges["translate_max"],
                size= (2,)
            ),
            scale = np.random.uniform(
                low= np.exp(-affine_ranges["scale_max"]),
                high= np.exp(affine_ranges["scale_max"]),
                size= (1,)
            ),
            shear = np.random.uniform(
                log= -affine_ranges["shear_max"],
                high= affine_ranges["shear_max"],
                size= (2,)
            ),
        )

        image = visionF.affine(image, **affine_kwargs)
        mask = visionF.affine(mask, **affine_kwargs)
        return image, mask

    def synth_videos(self, images, masks):
        """ Synthesize video clips by torch images. Return a torch.Tensor as a batch of
        video clips
        """
        pil_images = [self.to_pil_image(img) for img in images]
        pil_masks = [self.to_pil_image(msk) for msk in masks]
        videos, m_videos = [], []
        with torch.no_grad():
            for image, mask in zip(pil_images, pil_masks):
                video, m_video = [], []
                for frame_i in range(self.data_augment_kwargs["n_frames"]):
                    frame, m_frame = self.random_transforms(image, mask)
                    frame, m_frame = self.to_tensor(frame), self.to_tensor(m_frame)
                    video.append(frame)
                    m_video.append(m_frame)
                videos.append(torch.stack(video))
                m_videos.append(torch.stack(m_video))
        videos = torch.stack(videos)
        m_videos = torch.stack(m_videos)
        # the returned videos should be batch-wise
        return videos, m_videos
        
    def pretrain(self, epoch_i, data):
        """ As the paper described, pretrain on images is the first stage.
        @ Args:
            data: a dictionary with following keys
                "image": a torch.Tensor with size (b, C, H, W), usually C==3
                "mask": a torch.Tenwor with size (b, n, H, W) with one-hot encoding
                "n_objects": max number of objects
        """
        videos, masks = self.synth_videos(data["image"], data["mask"])

        self.optim.zero_grad()
        pred, loss = self.step(
            model_input= videos,
            ground_truth= masks,
            n_objects= data["n_objects"]
        )
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optim.step()

        return TrainInfo(loss= loss.detach().cpu().numpy(), gradNorm= grad_norm), \
            dict(
                videos= videos.cpu().numpy(),
                preds= pred.detach().cpu().numpy(),
                n_objects= data["n_objects"]
            )

    def train(self, epoch_i, data):
        """
        @ Args:
            data: a dictionary with following keys
                "video": a torch.Tensor with size (b, t, C, H, W), usually C == 3
                "mask": a torch.Tenwor with size (b, t, n, H, W) with one-hot encoding
                "n_objects": max number of objects
        """
        self.optim.zero_grad()
        pred, loss = self.step(
            model_input= data["video"],
            ground_truth= data["mask"],
            n_objects= data["n_objects"]
        )
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optim.step()

        return TrainInfo(loss= loss.detach().cpu().numpy(), gradNorm= grad_norm), \
            dict(
                videos= data["video"].cpu().numpy(),
                preds= pred.detach().cpu().numpy(),
                n_objects= data["n_objects"]
            )


    def eval(self, epoch_i, data):
        """
        @ Args:
            data: a dictionary with following keys
                "video": a torch.Tensor with size (b, t, C, H, W), usually C == 3
                "mask": a torch.Tenwor with size (b, t, n, H, W) with one-hot encoding
                "n_objects": max number of objects
        """
        with torch.no_grad():
            pred, loss = self.step(
                model_input= data["video"],
                ground_truth= data["mask"],
                n_objects= data["n_objects"]
            )
        return EvalInfo(loss= loss.cpu().numpy()), \
            dict(
                videos= data["video"].cpu().numpy(),
                preds= pred.cpu().numpy(),
                n_objects= data["n_objects"]
            )