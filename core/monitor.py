import numpy as np

from utils.module_loading import load_class

from .events import EventHandler
from .loggers import CSVLogger
from .metrics import Loss, Time, MetricsList


class Monitor(EventHandler):
    def __init__(self, config):
        self.metrics = MetricsList([Loss(), Time()])

        monitor_config = config.get('monitor')
        for metric_config in monitor_config:
            metric = load_class(metric_config)
            self.metrics.append(metric)

    def _on_epoch_end(self, state):
        self.metrics.update(state)
        metrics_data = self.metrics.get_data(state)
        state.update_epoch_results(**metrics_data)

    def _on_batch_start(self, state):
        self.metrics.reset_batch_data(state)

    def _on_batch_end(self, state):
        self.metrics.update_batch_data(state)

    def on_training_batch_end(self, state):
        self._on_batch_end(state)

    def on_training_batch_start(self, state):
        self._on_batch_start(state)

    def on_training_epoch_end(self, state):
        self._on_epoch_end(state)

    def on_validation_batch_end(self, state):
        self._on_batch_end(state)

    def on_validation_batch_start(self, state):
        self._on_batch_start(state)

    def on_validation_epoch_end(self, state):
        self._on_epoch_end(state)

    def on_test_batch_end(self, state):
        self._on_batch_end(state)

    def on_test_batch_start(self, state):
        self._on_batch_start(state)

    def on_test_epoch_end(self, state):
        self._on_epoch_end(state)

    def state_dict(self):
        return {'monitor': self.metrics.state_dict()}

    def load_state_dict(self, state_dict):
        self.metrics.load_state_dict(state_dict['monitor'])







