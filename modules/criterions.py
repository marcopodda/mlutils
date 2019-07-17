from torch import nn


class BCE(nn.Module):
    def __init__(self, **params):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return self.bce(outputs, targets.squeeze(-1))