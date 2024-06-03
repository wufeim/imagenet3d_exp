import numpy as np
import torch
import torch.nn as nn
import torchvision


class SwinTransPose(nn.Module):
    def __init__(self, backbone, output_dim):
        super().__init__()
        self.backbone = backbone
        self.output_dim = output_dim

        assert self.backbone in ['swin_t', 'swin_s', 'swin_b'], \
            f'Unsupported backbone {self.backbone} for SwinTransformer'

        self.model = torchvision.models.__dict__[self.backbone](pretrained=True)

        out_dim = 768

        self.pose_head = nn.Linear(out_dim, self.output_dim)
        self.pose_head.weight.data.normal_(mean=0.0, std=0.01)
        self.pose_head.bias.data.zero_()

    def forward(self, x):
        # https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py#L607
        x = self.model.features(x)
        x = self.model.norm(x)
        x = self.model.permute(x)
        x = self.model.avgpool(x)
        x = self.model.flatten(x)
        return self.pose_head(x)
