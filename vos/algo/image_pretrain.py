from vos.algo.base import AlgoBase, TrainInfo, EvalInfo
from vos.models.cross_entropy_onehot import CrossEntropyOneHot
from vos.utils.quick_args import save__init__args

from torchvision import transforms
import torch
from torch import nn
from torch.nn import functional as F

class ImagePretrainAlgo(AlgoBase):
    """ The algorithm re-producing STM training method
    https://arxiv.org/abs/1904.00607
    """
    def __init__(self,
            data_augment_kwargs= dict(
                methods_kwargs = dict(
                    affine= None,
                    zooming= None,
                    cropping= None,
                ),  # a dict of torchvision.transforms arguments.
                    # If None, this method will not be used.
                n_frames= 2, # 2 as minimum, usually 3
            ),
            clip_grad_norm= 1e9,
            loss_fn= CrossEntropyOneHot(),
            **kwargs,
        ):
        save__init__args(locals())
        super(ImagePretrainAlgo, self).__init__(self, loss_fn= loss_fn, **kwargs)

        # compose the image augmentation method
        transforms_methods = list()
        for k, v in self.data_augment_kwargs["methods_kwargs"].items():
            if k == "affine" and not v is None:
                transforms_methods.append(
                    transforms.RandomAffine(**v)
                )
            elif k == "cropping" and not v is None:
                transforms_methods.append(
                    transforms.RandomCrop(**v)
                )
        self.random_transforms = transforms.Compose(transforms_methods)
        # NOTE: the Compose is applied on PIL image of size (C, H, W), not torch.Tensor
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

    def synth_videos(self, images, masks):
        """ Synthesize video clips by torch images. Return a torch.Tensor as a batch of
        video clips
        """
        pil_images = [self.to_pil_image(img) for img in images]
        videos = []
        with torch.no_grad():
            for image in pil_images:
                video = []
                for frame_i in range(self.data_augment_kwargs["n_frames"]):
                    video.append(
                        self.to_tensor(self.random_transforms(image))
                    )
                videos.append(torch.stack(video))
        videos = torch.stack(videos)
        # the returned videos should be batch-wise
        return videos
        
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
                videos= videos,
                preds= pred,
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
                videos= data["video"],
                preds= pred,
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
                videos= data["video"],
                preds= pred,
            )