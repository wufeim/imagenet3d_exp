import torch
import torch.nn as nn


def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    def __init__(self, out_dim, use_n_blocks, use_avgpool, heads):
        super().__init__()
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.heads = heads

        self.linear_heads = nn.ModuleDict({str(i): nn.Linear(out_dim, h) for i, h in enumerate(self.heads)})
        for i in range(len(self.heads)):
            self.linear_heads[str(i)].weight.data.normal_(mean=0.0, std=0.01)
            self.linear_heads[str(i)].bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return [self.linear_heads[str(i)](output) for i in range(len(self.heads))]
