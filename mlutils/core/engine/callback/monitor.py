import numpy as np

from mlutils.core.event.handler import EventHandler
from mlutils.util.module_loading import load_class

from .loggers import CSVLogger
from .metrics import Loss, Time, MetricsList


class Monitor(EventHandler):
    def __init__(self, additional_metrics=[]):
        self.metrics = MetricsList([Loss()] + additional_metrics)

    def _on_batch_end(self, state):
        self.metrics.update_batch_data(state)

    def _on_epoch_start(self, state):
        self.metrics.reset_batch_data(state)
        state.init_epoch_results(self.metrics)

    def _on_epoch_end(self, state):
        self.metrics.update(state)
        state.update_epoch_results(self.metrics)

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







