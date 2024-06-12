# https://github.com/mbanani/probe3d/blob/main/evals/models/midas_final.py

from functools import partial
import types

import torch
import torch.nn as nn

from .linear_classifier import LinearClassifier, create_linear_input, select_layers


def midas_forward(self, x):
    h, w = x.shape[2:]
    assert h == 384 and w == 384

    x = self.patch_embed(x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.pos_embed

    x = self.norm_pre(x)

    embeds = []
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        embeds.append(x)

    embeds = [[x[:, 1:, :], x[:, 0, :]] for x in embeds]

    return embeds


class ProbeMiDaS(nn.Module):
    def __init__(self, backbone, heads, use_n_blocks_list, avgpool_list, lr_list, layers='last'):
        super().__init__()
        self.heads = heads
        self.use_n_blocks_list = use_n_blocks_list
        self.avgpool_list = avgpool_list
        self.lr_list = lr_list

        self.max_use_n_blocks = max(use_n_blocks_list)

        self.autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)

        assert backbone in ['DPT_Large']
        midas = torch.hub.load('intel-isl/MiDaS', backbone)
        self.model = midas.pretrained.model
        self.model.forward = types.MethodType(midas_forward, self.model)
        self.model.eval()

        sample_input = torch.zeros((2, 3, 384, 384))
        sample_x_tokens_list = self.model(sample_input)

        self.model.patch_size = 16
        num_layers = len(self.model.blocks)
        assert num_layers == len(sample_x_tokens_list)

        self.multilayers = select_layers(use_n_blocks_list, num_layers, layers)

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

        self.model.cuda()

    def forward(self, x):
        with torch.no_grad():
            x_tokens_list = self.model(x)
        return {k: v(x_tokens_list) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)
