import glob
import os

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from imagenet3d.utils import continuous_to_bin, call_func_by_name, IMAGENET_CATES


class IMAGENET3DPose(Dataset):
    def __init__(
        self,
        mode,
        root_path,
        transforms,
        multi_bin,
        image_processor=None,
        categories=None,
        **kwargs
    ):
        self.mode = mode
        self.root_path = root_path
        self.transforms = transforms
        self.multi_bin = multi_bin
        if image_processor is not None:
            self.image_processor = call_func_by_name(func_name=image_processor['class_name']+'.from_pretrained', **image_processor)
        else:
            self.image_processor = None
        if categories is None:
            self.categories = IMAGENET_CATES
        else:
            self.categories = categories
        self.categories = sorted(self.categories)

        self.img_path = os.path.join(root_path, mode, 'images')
        self.annot_path = os.path.join(root_path, mode, 'annotations')
        self.list_path = os.path.join(root_path, mode, 'lists')

        self.sample_list = []
        for cate_idx, cate in enumerate(self.categories):
            with open(os.path.join(self.list_path, cate, 'mesh01.txt'), 'r') as fp:
                samples = fp.read().strip().split('\n')
            self.sample_list += [(cate_idx, cate, s) for s in samples]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        cate_idx, cate, sample_name = self.sample_list[item]
        img_path = os.path.join(self.img_path, cate, sample_name+'.JPEG')
        annot_path = os.path.join(self.annot_path, cate, sample_name+'.npz')

        img = Image.open(img_path).convert('RGB')
        if self.image_processor is None:
            img = np.array(img)
        else:
            img = self.image_processor(images=img, return_tensors='pt').pixel_values[0]
        annot = dict(np.load(annot_path, allow_pickle=True))

        sample = {
            'name': sample_name,
            'img': img,
            'azimuth': float(annot['azimuth']),
            'elevation': float(annot['elevation']),
            'theta': float(annot['theta']),
            'distance': float(annot['distance']),
            'cate': cate_idx}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
