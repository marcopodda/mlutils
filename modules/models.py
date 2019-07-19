from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dim_input=10, dim_target=2, **params):
        super().__init__()
        self.dim_input = dim_input
        self.dim_layers = params['dim_layers']
        self.dim_target = dim_target

        self.input_layer = nn.Linear(self.dim_input, self.dim_layers[0])

        hidden_layers = []
        for i, _ in enumerate(self.dim_layers[1:], 1):
            hidden_layer = nn.Linear(self.dim_layers[i-1], self.dim_layers[i])
            hidden_layers.append(hidden_layer)
        self.hidden_layers = nn.ModuleList(hidden_layers)

        self.output_layer = nn.Linear(self.dim_layers[-1], self.dim_target)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        return self.output_layer(x)