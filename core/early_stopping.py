import operator
import numpy as np

from .events import EventHandler


class EarlyStopper:
    def __init__(self, metric):
        self.metric = metric

    def check_for_early_stop(self, phase, state):
        if phase == self.metric.early_stopping_on:
            data = self.metric.get_data(phase)
            stop = self._stop_criterion(data)
            state.update(stop_training=stop)

    def _stop_criterion(self, state_dict):
        raise NotImplementedError

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__ = state_dict


class GLEarlyStopper(EarlyStopper):
    def __init__(self, metric, alpha=5):
        super().__init__(metric=metric)
        self.alpha = alpha

    def _stop_criterion(self, data):
        best_value = data['best']
        current_value = data['current']

        if self.metric.greater_is_better:
            return (best_value / current_value - 1) > self.alpha

        return 100 * (current_value / best_value - 1) > self.alpha


class PatienceEarlyStopper(EarlyStopper):
    def __init__(self, metric, patience=20):
        super().__init__(metric=metric)
        self.patience = patience

    def _stop_criterion(self, data):
        current_epoch = len(data['values'])
        best_epoch = data['best_epoch']
        return current_epoch - best_epoch > self.patience