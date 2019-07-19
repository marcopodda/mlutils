import time
import numpy as np

import torch

from sklearn import metrics


class Metric:
    def __init__(self):
        assert hasattr(self, 'greater_is_better')
        self.training = None
        self.validation = None
        self.test = None

    def _update(self, phase, value):
        setattr(self, phase, value)

    def update(self, state):
        raise NotImplementedError

    def get_data(self, phase):
        return self.data[phase]

    def state_dict(self):
        return self.__dict__.copy()

    def load_state_dict(self, state_dict):
        self.__dict__ = state_dict

    def value(self, phase):
        return getattr(self, phase)


class Loss(Metric):
    name = "loss"
    greater_is_better = False

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
    greater_is_better = False

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
        assert hasattr(self, 'name')
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
        value = self.metric_fun(targets, outputs)
        return self._update(state.phase, value=value)

    def _prepare_data(self, outputs, targets):
        raise NotImplementedError


class BinaryAccuracy(PerformanceMetric):
    name = "accuracy"
    greater_is_better = True
    metric_fun = staticmethod(metrics.accuracy_score)

    def _prepare_data(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        return (outputs > 0.5).numpy(), (targets == 1).numpy()


class MSE(PerformanceMetric):
    name = "mse"
    metric_fun = staticmethod(metrics.mean_squared_error)
    greater_is_better = False

    def _prepare_data(self, outputs, targets):
        return torch.sigmoid(outputs), targets


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

    def get_data(self, state):
        data = {}
        for metric in self._metrics:
            data[f"{state.phase}_{metric.name}"] = metric.value(state.phase)
        return data

    def __getitem__(self, index):
        return self._metrics[index]

    def __len__(self):
        return len(self._metrics)

    def __iter__(self):
        return iter(self._metrics)

    def state_dict(self):
        return [m.state_dict() for m in self._metrics]

    def load_state_dict(self, state_dicts):
        for metric, state_dict in zip(self._metrics, state_dicts):
            metric.load_state_dict(state_dict)