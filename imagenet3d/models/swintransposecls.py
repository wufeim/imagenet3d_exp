import numpy as np
import torch
import torch.nn as nn
import torchvision


class SwinTransPoseCls(nn.Module):
    def __init__(self, backbone, pose_output_dim, cls_output_dim):
        super().__init__()
        self.backbone = backbone
        self.pose_output_dim = pose_output_dim
        self.cls_output_dim = cls_output_dim

        assert self.backbone in ['swin_t', 'swin_s', 'swin_b'], \
            f'Unsupported backbone {self.backbone} for SwinTransformer'

        self.model = torchvision.models.__dict__[self.backbone](pretrained=True)

        if backbone in ['swin_t', 'swin_s']:
            out_dim = 768
        else:
            out_dim = 1024

        self.cls_head = nn.Linear(out_dim, self.cls_output_dim)
        self.cls_head.weight.data.normal_(mean=0.0, std=0.01)
        self.cls_head.bias.data.zero_()

        self.pose_head = nn.Linear(out_dim, self.pose_output_dim)
        self.pose_head.weight.data.normal_(mean=0.0, std=0.01)
        self.pose_head.bias.data.zero_()

    def forward(self, x):
        # https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py#L607
        x = self.model.features(x)
        x = self.model.norm(x)
        x = self.model.permute(x)
        x = self.model.avgpool(x)
        x = self.model.flatten(x)
        return self.cls_head(x), self.pose_head(x)
