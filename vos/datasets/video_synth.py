import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as visionF
import numpy as np

from vos.utils.quick_args import save__init__args
from vos.utils.image_shaper import random_crop_CHW

class VideoSynthDataset(Dataset):
    """ A wrapper that synthesize image dataset to a video-like dataset by randomly jitter
    the given image. It is a video dataset.
    """
    def __init__(self, dataset,
            n_frames: int, # video length
            resolution= (384, 384), # output video resolution
            resize_method= "crop", # choose between "crop", "resize"
            affine_kwargs= dict(
                angle_max= 180.,
                translate_max= 50.,
                scale_max= 2., # NOTE: this is the exponent of e
                shear_max= 50.
            ), # a dict of kwargs providing for torchvision.transforms.functional.affine
        ):
        save__init__args(locals())
        self.to_pil_image = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def random_transforms(self, image, mask):
        """ randomly generate a transform arguments, and apply it to both image and mask.
        Considering torchvision transform can only transform a single image, in requires
        some meanuvers to do with mask.
            NOTE: both arguments are torh.Tensor
        """
        affine_ranges = self.affine_kwargs
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
        for m_i in range(mask.shape[0]): # for each channel
            msk = mask[m_i:m_i+1]
            if m_i == 0:
                # back ground channel is a bit different
                msk = ~msk
            jittered_m = visionF.affine(self.to_pil_image(msk), **affine_kwargs)
            j_mask = self.to_tensor(jittered_m) # dtype = torch.float32
            # Assuming there are only j_mask.max() and j_mask.min() two kinds of value in j_mask
            j_max = j_mask.max()
            if j_max == 0:
                j_max = 1 
            j_mask_binary = (j_mask == j_max).to(dtype= torch.uint8)
            # background is a bit different
            if m_i == 0:
                j_mask_binary = ~j_mask_binary
            layers_of_mask.extend([j_mask_binary])
        mask = torch.cat(layers_of_mask, 0).to(dtype= torch.uint8)

        return image, mask

    def synth_videos(self, images, masks):
        """ Synthesize video clips by torch images. Return a torch.Tensor as a batch of
        video clips
        """
        videos, m_videos = [], []
        with torch.no_grad():
            for image, mask in zip(images, masks):
                video, m_video = [image], [mask]
                for frame_i in range(self.n_frames-1):
                    frame, m_frame = self.random_transforms(image, mask)
                    video.append(frame)
                    m_video.append(m_frame)
                videos.append(torch.stack(video))
                m_videos.append(torch.stack(m_video))
        videos = torch.stack(videos)
        m_videos = torch.stack(m_videos)
        # the returned videos should be batch-wise
        return videos, m_videos
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        image = img["image"]
        mask = img["mask"]

        with torch.no_grad():
            video, m_video = self.synth_videos([image], [mask])
        video, m_video = video[0], m_video[0]

        if self.resize_method == "crop":
            cropped, _ = random_crop_CHW(self.resolution, (video, m_video))
            video, m_video = cropped
        elif self.resize_method == "resize":
            raise NotImplementedError # put here for later implementation
        else:
            raise NotImplementedError

        img.pop("image")
        img["video"] = video
        img["mask"] = m_video
        return img
