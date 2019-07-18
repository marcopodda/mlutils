import operator
import numpy as np

import torch

from sklearn import metrics


class Metric:
    def __init__(self, monitor_on=['training', 'validation', 'test'],
                       early_stopping_on='validation',
                       save_best_on='validation'):
        assert hasattr(self, 'greater_is_better')

        self.monitor_on = monitor_on
        self.early_stopping_on = early_stopping_on
        self.save_best_on = save_best_on

        self.data = {}
        for phase in self.monitor_on:
            best = -np.float('inf') if self.greater_is_better else np.float('inf')
            self.data.update({
                phase: {
                    'best': best,
                    'is_best': False,
                    'best_epoch': 0,
                    'current': None,
                    'values': []
                }
            })

    @classmethod
    def op(cls, value, best_value):
        if cls.greater_is_better:
            return operator.gt(value, best_value)
        return operator.lt(value, best_value)

    def _update(self, phase, value):
        if phase in self.monitor_on:
            self.data[phase]['current'] = value
            self.data[phase]['values'].append(value)

            if self.op(value, self.data[phase]['best']):
                current_epoch = len(self.data[phase]['values'])
                self.data[phase]['best'] = value
                self.data[phase]['is_best'] = True
                self.data[phase]['best_epoch'] = current_epoch
            else:
                self.data[phase]['is_best'] = False

    def update(self, phase, state):
        raise NotImplementedError

    def get_data(self, phase):
        return self.data[phase]

    def state_dict(self):
        return self.__dict__.copy()

    def load_state_dict(self, state_dict):
        self.__dict__ = state_dict


class Loss(Metric):
    greater_is_better = False

    def __init__(self, monitor_on=['training', 'validation', 'test'],
                       early_stopping_on='validation',
                       save_best_on='validation'):
        super().__init__(monitor_on=monitor_on)

        self.batch_losses = []

    def update_batch_data(self, state):
        loss = state.loss.item()
        self.batch_losses.append(loss)

    def reset_batch_data(self):
        self.batch_losses = []

    def update(self, phase):
        epoch_loss = sum(self.batch_losses) / len(self.batch_losses)
        return self._update(phase, value=epoch_loss)


class PerformanceMetric(Metric):
    def __init__(self, monitor_on=['training', 'validation', 'test'],
                       early_stopping_on='validation',
                       save_best_on='validation'):
        super().__init__(monitor_on=monitor_on)
        assert hasattr(self, 'metric_fun')

        self.outputs = []
        self.targets = []

    def update_batch_data(self, state):
        self.outputs.append(state.outputs.detach())
        self.targets.append(state.targets.detach())

    def reset_batch_data(self):
        self.outputs = []
        self.targets = []

    def update(self, phase):
        outputs = torch.cat(self.outputs)
        targets = torch.cat(self.targets)

        if targets.dim() - outputs.dim() > 0:
            targets = targets.squeeze(-1)
        elif outputs.dim() - targets.dim() < 0:
            outputs = outputs.squeeze(-1)

        outputs, targets = self._prepare_data(outputs, targets)
        value = self.metric_fun(targets, outputs)
        return self._update(phase, value=value)

    def _prepare_data(self, outputs, targets):
        raise NotImplementedError


class BinaryAccuracy(PerformanceMetric):
    greater_is_better = True
    metric_fun = staticmethod(metrics.accuracy_score)

    def _prepare_data(self, outputs, targets):
        outputs = torch.sigmoid(outputs)
        return (outputs > 0.5).numpy(), (targets == 1).numpy()


class MSE(PerformanceMetric):
    metric_fun = staticmethod(metrics.mean_squared_error)
    greater_is_better = False

    def _prepare_data(self, outputs, targets):
        return outputs, targets