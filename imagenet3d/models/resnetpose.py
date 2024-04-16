import numpy as np
import torch.nn as nn
import torchvision


class ResNetPose(nn.Module):
    def __init__(self, backbone, output_dim):
        super().__init__()
        self.backbone = backbone
        self.output_dim = output_dim

        assert self.backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], \
            f'Unsupported backbone {self.backbone} for PoseResNet'

        self.model = torchvision.models.__dict__[self.backbone](pretrained=True)
        self.model.avgpool = nn.AvgPool2d(8, stride=1)
        if self.backbone == 'resnet18':
            self.model.fc = nn.Linear(512 * 1, self.output_dim)
        else:
            self.model.fc = nn.Linear(512 * 4, self.output_dim)

    def forward(self, img):
        return self.model(img)
