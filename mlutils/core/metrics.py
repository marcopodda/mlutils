import time
import operator as op
import numpy as np

import torch
from torch.nn import functional as F

from sklearn import metrics


class Metric:
    def __init__(self):
        assert hasattr(self, 'name')
        self.training = None
        self.validation = None
        self.test = None

    @property
    def greater_is_better(self):
        return self.operator == op.gt

    def _update(self, phase, value):
        setattr(self, phase, value)

    def update(self, state):
        raise NotImplementedError

    def get_value(self, phase):
        return getattr(self, phase)


class MetricsList:
    def __init__(self, metrics):
        self._metrics = metrics

    def update_batch_data(self, state):
        for metric in self._metrics:
            metric.update_batch_data(state)

    def reset_batch_data(self, state):
        for metric in self._metrics:
            metric.reset_batch_data(state)

    def update(self, state):
        for metric in self._metrics:
            metric.update(state)

    def append(self, metric):
        self._metrics.append(metric)

    def get_data(self, phase):
        data = {}
        for metric in self._metrics:
            data[f"{phase}_{metric.name}"] = metric.get_value(phase)
        return data

    def __getitem__(self, index):
        return self._metrics[index]

    def __len__(self):
        return len(self._metrics)

    def __iter__(self):
        return iter(self._metrics)


class Loss(Metric):
    name = "loss"
    operator = op.lt

    def __init__(self):
        super().__init__()
        self.batch_losses = []

    def update_batch_data(self, state):
        loss = state.loss.item()
        self.batch_losses.append(loss)

    def reset_batch_data(self, state):
        self.batch_losses = []

    def update(self, state):
        epoch_loss = sum(self.batch_losses) / len(self.batch_losses)
        return self._update(state.phase, value=epoch_loss)


class Time(Metric):
    name = "time"
    operator = op.lt

    def __init__(self):
        super().__init__()
        self.batch_times = []

    def update_batch_data(self, state):
        self.batch_times[-1] = time.time() - self.batch_times[-1]

    def reset_batch_data(self, state):
        self.batch_times.append(time.time())

    def update(self, state):
        time_elapsed = sum(self.batch_times)
        self.batch_times = []
        return self._update(state.phase, value=time_elapsed)


class PerformanceMetric(Metric):
    def __init__(self):
        assert hasattr(self, 'metric_fun')
        super().__init__()

        self.outputs = []
        self.targets = []

    def update_batch_data(self, state):
        self.outputs.append(state.outputs.detach())
        self.targets.append(state.targets.detach())

    def reset_batch_data(self, state):
        self.outputs = []
        self.targets = []

    def update(self, state):
        outputs = torch.cat(self.outputs)
        targets = torch.cat(self.targets)

        if targets.dim() - outputs.dim() > 0:
            targets = targets.squeeze(-1)
        elif outputs.dim() - targets.dim() < 0:
            outputs = outputs.squeeze(-1)

        outputs, targets = self._prepare_data(outputs, targets)
        value = self.metric_fun(targets.numpy(), outputs.numpy())
        return self._update(state.phase, value=value)

    def _prepare_data(self, outputs, targets):
        raise NotImplementedError


class BinaryAccuracy(PerformanceMetric):
    name = "accuracy"
    metric_fun = staticmethod(metrics.accuracy_score)
    operator = op.gt

    def _prepare_data(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        return (outputs > 0.5), (targets == 1)


class MulticlassAccuracy(PerformanceMetric):
    name = "accuracy"
    metric_fun = staticmethod(metrics.accuracy_score)
    operator = op.gt

    def _prepare_data(self, outputs, targets):
        outputs = F.softmax(outputs, dim=-1)
        return outputs.argmax(dim=-1), targets


class MSE(PerformanceMetric):
    name = "mse"
    metric_fun = staticmethod(metrics.mean_squared_error)
    operator = op.lt

    def _prepare_data(self, outputs, targets):
        return outputs, targets


class MAE(PerformanceMetric):
    name = "mae"
    metric_fun = staticmethod(metrics.mean_absolute_error)
    operator = op.lt

    def _prepare_data(self, outputs, targets):
        return outputs, targets


class AUC(PerformanceMetric):
    name = "auc"
    metric_fun = staticmethod(metrics.roc_auc_score)
    operator = op.gt

    def _prepare_data(self, outputs, targets):
        return torch.sigmoid(outputs), targets