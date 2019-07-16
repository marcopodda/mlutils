import torch
from torch.nn import functional as F

from sklearn import metrics

from .events import EventHandler


class Metric:
    def __init__(self):
        self.values = []

    def update(self, outputs, targets):
        if targets.dim() - outputs.dim() > 0:
            targets = targets.squeeze(-1)
        elif outputs.dim() - targets.dim() < 0:
            outputs = outputs.squeeze(-1)

        outputs, targets = self._prepare_data(outputs, targets)
        self.values.append(self.fun(targets, outputs))

    def _prepare_data(self, outputs, targets):
        raise NotImplementedError

    def value(self):
        if self.values == []:
            return None
        return self.values[-1]


class BinaryAccuracy(Metric):
    fun = staticmethod(metrics.accuracy_score)

    def _prepare_data(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        return (outputs > 0.5).numpy(), (targets == 1).numpy()


class MulticlassAccuracy(Metric):
    fun = staticmethod(metrics.accuracy_score)

    def _prepare_data(self, outputs, targets):
        outputs = F.softmax(outputs, dim=-1)
        return outputs.argmax().numpy(), targets.numpy()


class MSE(Metric):
    fun = staticmethod(metrics.mean_squared_error)

    def _prepare_data(self, outputs, targets):
        return outputs, targets