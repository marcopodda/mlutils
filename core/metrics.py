import operator
import numpy as np

import torch
from torch.nn import functional as F

from sklearn import metrics

from .events import EventHandler


class BaseMetric:
    def __init__(self):
        self.values = []
        self.best = -np.float('inf') if self.greater_is_better else np.float('inf')
        self.best_epoch = 0
        self.op = operator.gt if self.greater_is_better else operator.lt

    def epoch_start(self, state):
        pass

    def epoch_end(self, state):
        pass

    def batch_start(self, state):
        pass

    def batch_end(self, state):
        pass

    def epoch_end(self, state):
        pass

    def _update(self, value):
        self.values.append(value)

        if self.op(value, self.best):
            self.best = value
            self.best_epoch = self.values.index(self.best)

        return {
            'is_best': self.best_epoch == len(self.values) - 1,
            'best': self.best,
            'best_epoch': self.best_epoch,
            'current': self.values[-1]
        }

    def to_dict(self):
        return {
            'values': self.values,
            'best': self.best,
            'best_epoch': self.best_epoch
        }

    def from_dict(self, state_dict):
        self.values = state_dict['values']
        self.best = state_dict['best']
        self.best_epoch = state_dict['best_epoch']


class ModelLoss(BaseMetric):
    greater_is_better = False

    def __init__(self):
        super().__init__()
        self.batch_losses = []

    def batch_start(self, state):
        self.batch_losses = []

    def batch_end(self, state):
        loss = state.loss.item()
        self.batch_losses.append(loss)

    def epoch_end(self, state):
        epoch_loss = sum(self.batch_losses) / len(self.batch_losses)
        return self._update(value=epoch_loss)



class Metric(BaseMetric):
    def __init__(self):
        super().__init__()
        self.outputs = []
        self.targets = []

    def _prepare_data(self, outputs, targets):
        raise NotImplementedError

    def epoch_start(self, state):
        self.outputs = []
        self.targets = []

    def batch_end(self, state):
        self.outputs.append(state.outputs.detach())
        self.targets.append(state.targets.detach())

    def epoch_end(self, state):
        outputs = torch.cat(self.outputs)
        targets = torch.cat(self.targets)

        if targets.dim() - outputs.dim() > 0:
            targets = targets.squeeze(-1)
        elif outputs.dim() - targets.dim() < 0:
            outputs = outputs.squeeze(-1)

        outputs, targets = self._prepare_data(outputs, targets)
        value = self.fun(targets, outputs)
        return self._update(value=value)


class BinaryAccuracy(Metric):
    fun = staticmethod(metrics.accuracy_score)
    greater_is_better = True

    def _prepare_data(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        return (outputs > 0.5).numpy(), (targets == 1).numpy()


class MulticlassAccuracy(Metric):
    fun = staticmethod(metrics.accuracy_score)
    greater_is_better = True

    def _prepare_data(self, outputs, targets):
        outputs = F.softmax(outputs, dim=-1)
        return outputs.argmax().numpy(), targets.numpy()


class MSE(Metric):
    fun = staticmethod(metrics.mean_squared_error)
    greater_is_better = False

    def _prepare_data(self, outputs, targets):
        return outputs, targets