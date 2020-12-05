import sys
import pickle
import os
import numpy as np

import copy
import PIL
import torch

class ImageList(torch.utils.data.Dataset):
    """
    Basic dataloader for images
    """

    def __init__(self, image_paths, preprocess=None):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with open(image_path, 'rb') as f:
            image = PIL.Image.open(f).convert('RGB')
        if self.preprocess is not None:
            image = self.preprocess(image)

        return image

    def __len__(self):
        return len(self.image_paths)


class FrameDataset(torch.utils.data.Dataset):

    def __init__(self, samples, image_dir, mode="CROP", preprocess=None):
        # note here preprocess does not include scaling!
        self.samples = samples
        self.image_dir = image_dir
        self.mode = mode
        self.preprocess = preprocess

    def __getitem__(self, index):
        ids = list(self.samples.keys())  # pedID/video_/frame
        idx = ids[index]
        frame = self.samples[idx]['frame']
        bbox = copy.deepcopy(self.samples[idx]['bbox'])
        source = self.samples["source"]
        label = self.samples[idx]['future_state']
        current_state = self.samples[idx]['current_state']
        image_path = None
        # image paths
        if source == "JAAD":
            vid = self.samples[idx]['video_number']
            image_path = os.path.join(self.image_dir, vid, '{:05d}.png'.format(frame))
        elif source == "PIE":
            vid = self.samples[idx]['video_number']
            sid = self.samples[idx]['set_number']
            image_path = os.path.join(self.image_dir, sid, vid, '{:05d}.png'.format(frame))
        with open(image_path, 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')

        if self.image_type == "CROP":
            if self.preprocess is not None:
                img, bbox = self.preprocess(img, bbox)

        elif self.image_type == "WHOLE":
            img, bbox = crop_and_rescale(img, bbox)
            if self.preprocess is not None:
                img = self.preprocess(img)

        img_tensor = torchvision.transforms.ToTensor()(img)
        label = torch.tensor(label)
        label = label.to(torch.float32)

        sample = {'image': img_tensor, 'bbox': bbox, 'current_state': current_state}

        return sample, label

    def __len__(self):
        return len(self.samples.keys())
