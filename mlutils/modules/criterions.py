import torch
from torch import nn


class LossFunction(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, outputs, targets):
        targets = self.reshape_targets(targets)
        return self.loss(outputs, targets)

    def reshape_targets(self, targets):
        raise NotImplementedError


class BinaryCrossEntropy(LossFunction):
    def __init__(self, **params):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def reshape_targets(self, targets):
        batch_size, dim_target = targets.size(0), targets.size(1)
        return targets.reshape(batch_size, dim_target)


class CrossEntropy(LossFunction):

    def __init__(self, **params):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def reshape_targets(self, targets):
        batch_size, dim_target = targets.size(0), targets.size(1)
        return targets.reshape(batch_size)


class MeanSquaredError(LossFunction):
    def __init__(self, **params):
        super().__init__()
        self.loss = nn.MSELoss()

    def reshape_targets(self, targets):
        batch_size, dim_target = targets.size(0), targets.size(1)
        return targets.reshape(batch_size, dim_target)