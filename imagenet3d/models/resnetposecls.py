import numpy as np
import torch
import torch.nn as nn
import torchvision


class ResNetPoseCls(nn.Module):
    def __init__(self, backbone, pose_output_dim, cls_output_dim):
        super().__init__()
        self.backbone = backbone
        self.pose_output_dim = pose_output_dim
        self.cls_output_dim = cls_output_dim

        assert self.backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], \
            f'Unsupported backbone {self.backbone} for PoseResNet'

        self.model = torchvision.models.__dict__[self.backbone](pretrained=True)
        self.model.avgpool = nn.AvgPool2d(8, stride=1)
        if self.backbone == 'resnet18':
            self.model.fc = nn.Linear(512 * 1, self.pose_output_dim + self.cls_output_dim)
        else:
            self.model.fc = nn.Linear(512 * 4, self.pose_output_dim + self.cls_output_dim)

        out_dim = 512 * 1 if self.backbone == 'resnet18' else 512 * 4

        self.cls_head = nn.Linear(out_dim, self.cls_output_dim)
        self.cls_head.weight.data.normal_(mean=0.0, std=0.01)
        self.cls_head.bias.data.zero_()

        self.pose_head = nn.Linear(out_dim, self.pose_output_dim)
        self.pose_head.weight.data.normal_(mean=0.0, std=0.01)
        self.pose_head.bias.data.zero_()

    def forward(self, x):
        # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L266
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return self.cls_head(x), self.pose_head(x)
