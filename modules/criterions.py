from torch import nn


class BCE(nn.Module):
    def __init__(self, config, dim_target):
        super().__init__()
        self.config = config
        self.dim_target = dim_target

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return self.bce(outputs, targets.squeeze(-1))