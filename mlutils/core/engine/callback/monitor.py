import numpy as np

from mlutils.core.event.handler import EventHandler
from mlutils.core.engine.callback.metrics import MetricsList, Loss


class Monitor(EventHandler):
    def __init__(self, additional_metrics=[]):
        self.metrics = MetricsList([Loss()] + additional_metrics)

    def _on_batch_end(self, state):
        self.metrics.update_batch_data(state)

    def _on_epoch_start(self, state):
        self.metrics.reset_batch_data(state)

        for metric in self.metrics:
            key = f"{state.phase}_{metric.name}"

            if key not in state.epoch_results:
                state.epoch_results.update(**{key: {}})

            if key not in state.best_results:
                best = -np.float('inf') if metric.greater_is_better else np.float('inf')
                best_dict = {'best': best, 'best_epoch': 0, 'prev_best': best, 'prev_best_epoch': 0}
                state.best_results.update(**{key: best_dict})

    def _on_epoch_end(self, state):
        self.metrics.update(state)

        for metric in self.metrics:
            key = f"{state.phase}_{metric.name}"
            state.epoch_results.update(**self.metrics.get_data(state.phase))
            best_value = state.best_results[key]['best']
            current_value = metric.get_value(state.phase)
            if metric.op(current_value, best_value):
                state.best_results[key]['prev_best'] = state.best_results[key]['best']
                state.best_results[key]['prev_best_epoch'] = state.best_results[key]['best_epoch']
                state.best_results[key]['best'] = current_value
                state.best_results[key]['best_epoch'] = state.epoch

    def on_training_batch_end(self, state):
        self._on_batch_end(state)

    def on_validation_batch_end(self, state):
        self._on_batch_end(state)

    def on_test_batch_end(self, state):
        self._on_batch_end(state)

    def on_training_epoch_start(self, state):
        self._on_epoch_start(state)

    def on_validation_epoch_start(self, state):
        self._on_epoch_start(state)

    def on_test_epoch_start(self, state):
        self._on_epoch_start(state)

    def on_training_epoch_end(self, state):
        self._on_epoch_end(state)

    def on_validation_epoch_end(self, state):
        self._on_epoch_end(state)

    def on_test_epoch_end(self, state):
        self._on_epoch_end(state)

