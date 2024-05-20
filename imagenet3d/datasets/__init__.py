import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from imagenet3d.utils import construct_class_by_name, continuous_to_bin


class Resize:
    def __init__(self, height=256, width=256):
        self.transform = transforms.Resize(size=(height, width), antialias=True)

    def __call__(self, sample):
        sample['img'] = self.transform(sample['img'])
        return sample


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.transform = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        sample['img'] = self.transform(sample['img'])
        return sample


def hflip(sample):
    sample['img'] = transforms.functional.hflip(sample['img'])
    sample['azimuth'] = 2 * np.pi - sample['azimuth']
    sample['theta'] = - sample['theta']
    return sample


class RandomHorizontalFlip:
    def __init__(self):
        self.transform = transforms.RandomApply([lambda x: hflip(x)], p=0.5)

    def __call__(self, sample):
        return self.transform(sample)


class ToTensor:
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        if not isinstance(sample['img'], torch.Tensor):
            sample['img'] = self.transform(sample['img'])
        for k in ['azimuth', 'elevation', 'theta']:
            if k in sample:
                sample[k] = torch.tensor([sample[k]])[0]
        for k in ['azimuth_idx', 'elevation_idx', 'theta_idx', 'cate']:
            if k in sample:
                sample[k] = torch.tensor([sample[k]], dtype=torch.long)[0]
        return sample


class PoseToLabel:
    def __init__(self, num_bins, min_value, max_value, border_type='periodic'):
        self.num_bins = num_bins
        self.min_value = min_value
        self.max_value = max_value
        self.border_type = border_type

    def __call__(self, sample):
        for k in ['azimuth', 'elevation', 'theta']:
            if k in sample:
                sample[k+'_idx'] = continuous_to_bin(
                    sample[k], num_bins=self.num_bins, min_value=self.min_value, max_value=self.max_value)
        return sample


def build_dataset(data_cfg, mode='train', **dataset_kwargs):
    if data_cfg.transform_list is None or not data_cfg.transform_list:
        transform = None
    else:
        transform = transforms.Compose([
            construct_class_by_name(**t) for t in data_cfg.transform_list])

    dataset = construct_class_by_name(
        **data_cfg, transforms=transform, mode=mode, **dataset_kwargs)

    return dataset


def build_dataloader_train(cfg, **dataset_kwargs):
    train_dataset = build_dataset(cfg.data.train, 'train', **dataset_kwargs)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.workers)
    return train_dataloader


def build_dataloader_val(cfg, **dataset_kwargs):
    val_dataset = build_dataset(cfg.data.val, 'val', **dataset_kwargs)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.evaluate.batch_size,
        shuffle=False,
        num_workers=cfg.training.workers)
    return val_dataloader


def build_dataloader(cfg, **dataset_kwargs):
    train_dataloader = build_dataloader_train(cfg, **dataset_kwargs)
    val_dataloader = build_dataloader_val(cfg, **dataset_kwargs)
    return train_dataloader, val_dataloader
