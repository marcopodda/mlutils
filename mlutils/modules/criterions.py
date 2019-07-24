from torch import nn


class LossFunction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, outputs, targets):
        targets = self.reshape_targets(targets)

        return self.loss(outputs, targets)

    def reshape_targets(self, targets):
        raise NotImplementedError


class BinaryCrossEntropy(LossFunction):
    def __init__(self, config):
        super().__init__(config)
        self.loss = nn.BCEWithLogitsLoss(**config.params)

    def reshape_targets(self, targets):
        batch_size, dim_target = targets.size(0), targets.size(1)
        return targets.reshape(batch_size, dim_target)


class CrossEntropy(LossFunction):

    def __init__(self, config):
        super().__init__(config)
        self.loss = nn.CrossEntropyLoss(**config.params)

    def reshape_targets(self, targets):
        batch_size, _ = targets.size(0), targets.size(1)
        return targets.reshape(batch_size)


class MeanSquaredError(LossFunction):
    def __init__(self, config):
        super().__init__(config)
        self.loss = nn.MSELoss(**config.params)

    def reshape_targets(self, targets):
        batch_size, dim_target = targets.size(0), targets.size(1)
        return targets.reshape(batch_size, dim_target)
