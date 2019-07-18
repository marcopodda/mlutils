import operator
import numpy as np

from .events import EventHandler


class EarlyStopper(EventHandler):
    def __init__(self, phase, metric, greater_is_better):
        self.phase = phase
        self.metric = metric
        self.best_value = -np.float('inf') if greater_is_better else np.float('inf')
        self.op = operator.gt if greater_is_better else operator.lt

    def stop_condition(self, **kwargs):
        return False

    def on_training_epoch_end(self, state):
        if self.phase == 'training':
            self._on_epoch_end(state)

    def on_validation_epoch_end(self, state):
        if self.phase == 'validation':
            self._on_epoch_end(state)



class GLEarlyStopper(EarlyStopper):
    def __init__(self, alpha=5, phase='validation', metric='model_loss', greater_is_better=True):
        super().__init__(phase=phase, metric=metric, greater_is_better=greater_is_better)
        self.alpha = alpha

    def _on_epoch_end(self, state):
        metric = state[f"{self.phase}_monitor"]["metric"]
        current_value = metric['current']

        if self.op(current_value, self.best_value):
            self.best_value = current_value

        if self.greater_is_better:
            stop_condition = (self.best_value / current_value - 1) > self.alpha
        else:
            stop_condition = 100 * (current_value / self.best_value - 1) > self.alpha

        state.update(stop_training=stop_condition)


class PatienceEarlyStopper(EarlyStopper):
    def __init__(self, patience=20, phase='validation', metric='model_loss', greater_is_better=True):
        super().__init__(phase=phase, metric=metric, greater_is_better=greater_is_better)
        self.patience = patience

    def _on_epoch_end(self, state):
        metric = state[f"{self.phase}_{self.metric}"]
        current_value = metric['current']
        best_epoch = metric['best_epoch']

        if self.op(current_value, self.best_value):
            self.best_value = current_value

        stop_condition = (state.epoch - best_epoch) > self.patience
        state.update(stop_training=stop_condition)