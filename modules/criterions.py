from torch import nn


class BCE(nn.Module):
    def __init__(self, config, state_dict={}, device='cpu', **kwargs):
        super().__init__()
        self.config = config
        self.bce = nn.BCEWithLogitsLoss()
        self.device = device

        if state_dict != {}:
            self.load_state_dict(state_dict)

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return self.bce(outputs, targets.squeeze(-1))