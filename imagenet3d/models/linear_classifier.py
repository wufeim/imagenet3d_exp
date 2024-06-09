import torch
import torch.nn as nn


def select_layers(use_n_blocks_list, num_layers):
    multilayers = []
    for nb in use_n_blocks_list:
        select = nb.split('_')[1]
        nb = int(nb.split('_')[0])
        if nb == 1:
            multilayers.append([num_layers-1])
        elif nb == 4:
            if select == 'uniform':
                multilayers.append([num_layers//4-1, num_layers//2-1, num_layers//4*3-1, num_layers-1])
            elif select == 'last':
                multilayers.append([num_layers-4, num_layers-3, num_layers-2, num_layers-1])
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError
    return multilayers


def create_linear_input(x_tokens_list, multilayers, use_avgpool):
    intermediate_output = [x_tokens_list[idx] for idx in multilayers]
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
    def __init__(self, out_dim, multilayers, use_avgpool, heads):
        super().__init__()
        self.out_dim = out_dim
        self.multilayers = multilayers
        self.use_avgpool = use_avgpool
        self.heads = heads

        self.linear_heads = nn.ModuleDict({str(i): nn.Linear(out_dim, h) for i, h in enumerate(self.heads)})
        for i in range(len(self.heads)):
            self.linear_heads[str(i)].weight.data.normal_(mean=0.0, std=0.01)
            self.linear_heads[str(i)].bias.data.zero_()

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.multilayers, self.use_avgpool)
        return [self.linear_heads[str(i)](output) for i in range(len(self.heads))]
