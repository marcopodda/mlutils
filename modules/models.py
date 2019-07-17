from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dim_input, dim_target, **params):
        super().__init__()
        self.dim_input = dim_input
        self.dim_hidden = params['dim_hidden']
        self.dim_target = dim_target

        self.linear1 = nn.Linear(self.dim_input, self.dim_hidden)
        self.linear2 = nn.Linear(self.dim_hidden, self.dim_hidden)
        self.linear3 = nn.Linear(self.dim_hidden, self.dim_target)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)