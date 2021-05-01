import os
import copy
import PIL
import torch
import torchvision


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

    def __init__(self, samples, image_dir, preprocess=None):
        self.samples = samples
        self.image_dir = image_dir
        self.preprocess = preprocess

    def __getitem__(self, index):
        ids = list(self.samples.keys())
        idx = ids[index]
        frame = self.samples[idx]['frame']
        bbox = copy.deepcopy(self.samples[idx]['bbox'])
        source = self.samples[idx]["source"]
        anns = {'bbox': bbox, 'source': source}
        if 'trans_label' in list(self.samples[idx].keys()):
            label = self.samples[idx]['trans_label']
        else:
            label = None
        image_path = None
        # image paths
        if source == "JAAD":
            vid = self.samples[idx]['video_number']
            image_path = os.path.join(self.image_dir, 'JAAD', vid, '{:05d}.png'.format(frame))
        elif source == "PIE":
            vid = self.samples[idx]['video_number']
            sid = self.samples[idx]['set_number']
            image_path = os.path.join(self.image_dir, 'PIE', sid, vid, '{:05d}.png'.format(frame))
        elif source == "TITAN":
            vid = self.samples[idx]['video_number']
            image_path = os.path.join(self.image_dir, 'TITAN', vid, 'images', '{:06}.png'.format(frame))

        with open(image_path, 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')
        if self.preprocess is not None:
            img, anns = self.preprocess(img, anns)
        img_tensor = torchvision.transforms.ToTensor()(img)
        if label is not None:
            label = torch.tensor(label)
            label = label.to(torch.float32)
        sample = {'image': img_tensor, 'bbox': anns['bbox'], 'id': idx, 'label': label}

        return sample

    def __len__(self):
        return len(self.samples.keys())


class SequenceDataset(torch.utils.data.Dataset):
    """
    Basic dataloader for loading sequence/history samples
    """

    def __init__(self, samples, image_dir, preprocess=None):
        """
        :params: samples: transition history samples(dict)
                image_dir: root dir for images extracted from video clips
                preprocess: optional preprocessing on image tensors and annotations
        """
        self.samples = samples
        self.image_dir = image_dir
        self.preprocess = preprocess

    def __getitem__(self, index):
        ids = list(self.samples.keys())
        idx = ids[index]
        frames = self.samples[idx]['frame']
        bbox = copy.deepcopy(self.samples[idx]['bbox'])
        source = self.samples[idx]["source"]
        action = self.samples[idx]['action']
        anns = {'bbox': bbox, 'source': source}
        if 'trans_label' in list(self.samples[idx].keys()):
            label = self.samples[idx]['trans_label']
        else:
            label = None
        image_path = None
        # image paths
        img_tensors = []
        for i in range(len(frames)):
            if source == "JAAD":
                vid = self.samples[idx]['video_number']
                image_path = os.path.join(self.image_dir, 'JAAD', vid, '{:05d}.png'.format(frames[i]))
            elif source == "PIE":
                vid = self.samples[idx]['video_number']
                sid = self.samples[idx]['set_number']
                image_path = os.path.join(self.image_dir, 'PIE', sid, vid, '{:05d}.png'.format(frames[i]))
            elif source == "TITAN":
                vid = self.samples[idx]['video_number']
                image_path = os.path.join(self.image_dir, 'TITAN', vid, 'images', '{:06}.png'.format(frames[i]))
            with open(image_path, 'rb') as f:
                img = PIL.Image.open(f).convert('RGB')
            if self.preprocess is not None:
                img, anns = self.preprocess(img, anns)
            img_tensors.append(torchvision.transforms.ToTensor()(img))
        img_tensors = torch.stack(img_tensors)
        if label is not None:
            label = torch.tensor(label)
            label = label.to(torch.float32)

        sample = {'image': img_tensors, 'bbox': bbox, 'action': action, 'id': idx, 'label': label}

        return sample

    def __len__(self):
        return len(self.samples.keys())


class PaddedSequenceDataset(torch.utils.data.Dataset):
    """
    Basic dataloader for loading padded sequence/history samples
    """

    def __init__(self, samples, image_dir, padded_length=10, preprocess=None, hflip_p=0.5):
        """
        :params: samples: transition history samples(dict)
                image_dir: root dir for images extracted from video clips
                preprocess: optional preprocessing on image tensors and annotations
        """
        self.samples = samples
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.padded_length = padded_length
        self.hflip_p = hflip_p

    def __getitem__(self, index):
        ids = list(self.samples.keys())
        idx = ids[index]
        frames = self.samples[idx]['frame']
        bbox = copy.deepcopy(self.samples[idx]['bbox'])
        source = self.samples[idx]["source"]
        action = self.samples[idx]['action']
        TTE = round(self.samples[idx]["TTE"], 2)
        if 'trans_label' in list(self.samples[idx].keys()):
            label = self.samples[idx]['trans_label']
        else:
            label = None
        bbox_new = []
        image_path = None
        # image paths
        img_tensors = []
        hflip = True if float(torch.rand(1).item()) < self.hflip_p else False
        for i in range(len(frames)):
            anns = {'bbox': bbox[i], 'source': source}
            if source == "JAAD":
                vid = self.samples[idx]['video_number']
                image_path = os.path.join(self.image_dir['JAAD'], vid, '{:05d}.png'.format(frames[i]))
            elif source == "PIE":
                vid = self.samples[idx]['video_number']
                sid = self.samples[idx]['set_number']
                image_path = os.path.join(self.image_dir['PIE'], sid, vid, '{:05d}.png'.format(frames[i]))
            elif source == "TITAN":
                vid = self.samples[idx]['video_number']
                image_path = os.path.join(self.image_dir['TITAN'], vid, 'images', '{:06}.png'.format(frames[i]))
            with open(image_path, 'rb') as f:
                img = PIL.Image.open(f).convert('RGB')
            if hflip:
                # flip image and bbox
                img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                w, h = img.size
                x_max = w - anns['bbox'][0]
                x_min = w - anns['bbox'][2]
                anns['bbox'][0] = x_min
                anns['bbox'][2] = x_max
            if self.preprocess is not None:
                img, anns = self.preprocess(img, anns)
            img_tensors.append(torchvision.transforms.ToTensor()(img))
            bbox_new.append(anns['bbox'])
        img_tensors = torch.stack(img_tensors)
        imgs_size = img_tensors.size()
        img_tensors_padded = torch.zeros((self.padded_length, imgs_size[1], imgs_size[2], imgs_size[3]))
        img_tensors_padded[:imgs_size[0], :, :, :] = img_tensors
        bbox_new_padded = copy.deepcopy(bbox_new)
        action_padded = copy.deepcopy(action)
        for i in range(imgs_size[0], self.padded_length):
            bbox_new_padded.append([0, 0, 0, 0])
            action_padded.append(-1)
        # seq_len = torch.squeeze(torch.LongTensor(imgs_size[0]))
        seq_len = imgs_size[0]
        if label is not None:
            label = torch.tensor(label)
            label = label.to(torch.float32)

        sample = {'image': img_tensors_padded, 'bbox': bbox_new_padded, 'seq_length': seq_len, 'action': action_padded,
                  'id': idx, 'label': label,
                  'source': source, 'TTE': TTE}

        return sample

    def __len__(self):
        return len(self.samples.keys())
