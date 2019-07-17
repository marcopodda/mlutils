from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, config, dim_input, dim_target, state_dict={}, device='cpu', **kwargs):
        super().__init__()
        self.config = config
        self.dim_input = dim_input
        self.dim_target = dim_target
        self.device = device

        self.linear1 = nn.Linear(dim_input, config.dim_hidden)
        self.linear2 = nn.Linear(config.dim_hidden, config.dim_hidden)
        self.linear3 = nn.Linear(config.dim_hidden, dim_target)

        if state_dict != {}:
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)