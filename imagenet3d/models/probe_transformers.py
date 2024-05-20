from functools import partial

import torch
import torch.nn as nn

from .linear_classifier import LinearClassifier, create_linear_input


class ProbeTransformers(nn.Module):
    def __init__(self, backbone, heads, use_n_blocks_list, avgpool_list, lr_list):
        super().__init__()
        self.heads = heads
        self.use_n_blocks_list = use_n_blocks_list
        self.avgpool_list = avgpool_list
        self.lr_list = lr_list

        self.max_use_n_blocks = max(use_n_blocks_list)

        self.autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.float)

        if 'mae' in backbone:
            from transformers import ViTMAEModel
            self.backbone = ViTMAEModel.from_pretrained(backbone)
        elif 'dinov2' in backbone:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(backbone)
        elif 'clip' in backbone:
            from transformers import CLIPVisionModel
            self.backbone = CLIPVisionModel.from_pretrained(backbone)
        elif 'sam' in backbone:
            from transformers import SamModel
            model = SamModel.from_pretrained(backbone)
        else:
            raise ValueError(f'Backbone {backbone} not recognized')
        self.backbone.eval()

        sample_input = torch.zeros((2, 3, 224, 224))
        sample_x_tokens_list = self.get_intermediate_layers(sample_input)

        self.classifiers_dict = nn.ModuleDict()
        self.optim_param_groups = []
        for nb in use_n_blocks_list:
            for avgpool in avgpool_list:
                for lr in lr_list:
                    sample_output = create_linear_input(sample_x_tokens_list, nb, avgpool)
                    lc = LinearClassifier(sample_output.shape[-1], nb, avgpool, heads=self.heads)
                    lc.cuda()
                    self.classifiers_dict[f'linear_cls_blocks_{nb}_avgpool_{avgpool}_lr_{lr:.5f}'.replace('.', ',')] = lc
                    self.optim_param_groups.append({'params': lc.parameters(), 'lr': lr})

        self.backbone.cuda()

    def get_intermediate_layers(self, x):
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.backbone(x, output_hidden_states=True)['hidden_states'][-self.max_use_n_blocks:]
        features = [[x[:, 1:, :], x[:, 0, :]] for x in features]
        return features

    def forward(self, x):
        x_tokens_list = self.get_intermediate_layers(x)
        return {k: v(x_tokens_list) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)
