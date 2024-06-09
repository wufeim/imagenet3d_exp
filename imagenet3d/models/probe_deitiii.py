from functools import partial

import torch
import torch.nn as nn

from .deit_utils import deit_base_patch16_LS, deit_large_patch16_LS
from .linear_classifier import LinearClassifier, create_linear_input, select_layers


class ProbeDeiTIII(nn.Module):
    def __init__(self, backbone, heads, use_n_blocks_list, avgpool_list, lr_list):
        super().__init__()
        self.heads = heads
        self.use_n_blocks_list = use_n_blocks_list
        self.avgpool_list = avgpool_list
        self.lr_list = lr_list

        self.max_use_n_blocks = max(use_n_blocks_list)

        self.autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)

        assert backbone in ['deitiii_base', 'deitiii_large']
        img_size = 384
        if backbone == 'deitiii_base':
            self.backbone = deit_base_patch16_LS(True, img_size, True)
        elif backbone == 'deitiii_large':
            self.backbone = deit_large_patch16_LS(True, img_size, True)
        self.backbone.eval()

        self.patch_size = 16
        self.embed_size = (img_size / self.patch_size, img_size, self.patch_size)
        self.backbone.patch_embed.strict_img_size = False

        num_layers = len(self.backbone.blocks)
        feat_dim = self.backbone.num_features

        self.multilayers = select_layers(use_n_blocks_list, num_layers)

        sample_input = torch.zeros((2, 3, 384, 384))
        sample_x_tokens_list = self.get_intermediate_layers(sample_input)

        self.classifiers_dict = nn.ModuleDict()
        self.optim_param_groups = []
        for nb, ml in zip(use_n_blocks_list, self.multilayers):
            for avgpool in avgpool_list:
                for lr in lr_list:
                    sample_output = create_linear_input(sample_x_tokens_list, ml, avgpool)
                    lc = LinearClassifier(sample_output.shape[-1], ml, avgpool, heads=self.heads)
                    lc.cuda()
                    self.classifiers_dict[f'linear_cls_blocks_{nb}_avgpool_{avgpool}_lr_{lr:.5f}'.replace('.', ',')] = lc
                    self.optim_param_groups.append({'params': lc.parameters(), 'lr': lr})

        self.backbone.cuda()

    def get_intermediate_layers(self, x):
        with torch.inference_mode():
            with self.autocast_ctx():
                B, _, h, w = x.shape
                h, w = h // self.patch_size, w // self.patch_size

                x = self.backbone.patch_embed(x)
                cls_tokens = self.backbone.cls_token.expand(B, -1, -1)
                x = x + self.backbone.pos_embed
                x = torch.cat((cls_tokens, x), dim=1)

                features = []
                for i, blk in enumerate(self.backbone.blocks):
                    x = blk(x)
                    features.append(x)
        features = [[x[:, 1:, :], x[:, 0, :]] for x in features]
        return features

    def forward(self, x):
        with torch.no_grad():
            x_tokens_list = self.get_intermediate_layers(x)
        return {k: v(x_tokens_list) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)
