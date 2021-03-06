import operator
import numpy as np

from mlutils.core.event.handler import EventHandler
from mlutils.core.logging import Logger


class EarlyStopper(EventHandler):
    def __init__(self, monitor, mode, patience):
        self.monitor = monitor
        self.mode = self.set_mode(monitor)
        self.patience = patience
        self.count = 1
        self.op = operator.lt if self.mode == "min" else operator.gt
        self.logger = Logger()

    def set_mode(self, monitor):
        if 'loss' in monitor:
            return 'min'
        if 'acc' in monitor:
            return 'max'
        self.logger.warning("Inferring mode for early stopping: 'min'.")
        return 'min'


class PatienceEarlyStopper(EarlyStopper):
    def __init__(self, monitor="validation_loss", mode="min", patience=30):
        super().__init__(monitor, mode, patience)
        self.patience = patience

    def on_epoch_end(self, state):
        best_epoch = state.best_results[self.monitor]['best_epoch']
        state.stop_training = state.epoch - best_epoch >= self.patience
        if state.stop_training is True:
            self.logger.warning(f"Early stopping: epoch {state.epoch} - best epoch {best_epoch} - patience {self.patience}")


class GLEarlyStopper(EarlyStopper):
    def __init__(self, monitor="validation_loss", mode="min", patience=10, alpha=5):
        super().__init__(monitor, mode, patience)
        self.alpha = alpha

    def on_epoch_end(self, state):
        current_value = state.epoch_results[self.monitor]
        best_value = state.best_results[self.monitor]['prev_best']

        if self.mode == 'min':
            delta = (current_value / best_value - 1)
        else:
            delta = (best_value / current_value - 1) > self.alpha

        self.count = 1 if delta < self.alpha else self.count + 1
        state.stop_training = self.count >= self.patience

        if state.stop_training is True:
            self.logger.warning(f"Early stopping: epoch {state.epoch} - count {self.count} - patience {self.patience}")


class DeltaEarlyStopper(PatienceEarlyStopper):
    def __init__(self, monitor="validation_accuracy", mode="max", patience=5, min_delta=1e-5):
        super().__init__(monitor, mode, patience)
        self.min_delta = min_delta

    def on_epoch_end(self, state):
        current_value = state.epoch_results[self.monitor]
        best_value = state.best_results[self.monitor]['prev_best']

        if self.op(current_value, best_value):
            delta = np.abs(current_value - best_value)
            self.count = 1 if delta > self.min_delta else self.count + 1
        else:
            self.count += 1

        state.stop_training = self.count >= self.patience

        if state.stop_training is True:
            self.logger.warning(f"Early stopping: epoch {state.epoch} - count {self.count} - patience {self.patience}")
