import operator
import numpy as np

import torch

from .events import EventHandler
from utils.module_loading import import_string


def load_metric(name):
    name = "".join([n.capitalize() for n in name.split("_")])
    metric_class = import_string(f"core.metrics.{name}")
    return metric_class()


class Monitor(EventHandler):
    def __init__(self, config):
        super().__init__()

        self.monitor = {}
        for phase in ['training', 'validation', 'test']:
            if phase in config:
                self.monitor[phase] = {}
                for metric_name in config[phase]:
                    self.monitor[phase][metric_name] = load_metric(metric_name)

    def _on_epoch_start(self, phase, state):
        if phase in self.monitor:
            for metric_name in self.monitor[phase]:
                self.monitor[phase][metric_name].epoch_start(state)

    def _on_batch_start(self, phase, state):
        if phase in self.monitor:
            for metric_name in self.monitor[phase]:
                self.monitor[phase][metric_name].batch_start(state)

    def _on_batch_end(self, phase, state):
        if phase in self.monitor:
            for metric_name in self.monitor[phase]:
                metric = self.monitor[phase][metric_name]
                metric.batch_end(state)

    def _on_epoch_end(self, phase, state):
        if phase in self.monitor:
            for metric_name in self.monitor[phase]:
                metric =  self.monitor[phase][metric_name]
                metric_state = metric.epoch_end(state)
                state.update(**{f"{phase}_{metric_name}": metric_state})

    def on_training_epoch_start(self, state):
        self._on_epoch_start('training', state)

    def on_validation_epoch_start(self, state):
        self._on_epoch_start('validation', state)

    def on_test_start(self, state):
        self._on_epoch_start('test', state)

    def on_training_batch_start(self, state):
        self._on_batch_start('training', state)

    def on_validation_batch_start(self, state):
        self._on_batch_start('validation', state)

    def on_test_batch_start(self, state):
        self._on_batch_start('test', state)

    def on_training_batch_end(self, state):
        self._on_batch_end('training', state)

    def on_validation_batch_end(self, state):
        self._on_batch_end('validation', state)

    def on_test_batch_end(self, state):
        self._on_batch_end('test', state)

    def on_training_epoch_end(self, state):
        self._on_epoch_end('training', state)

    def on_validation_epoch_end(self, state):
        self._on_epoch_end('validation', state)

    def on_test_end(self, state):
        self._on_epoch_end('test', state)

    def state_dict(self):
        state_dict = {}
        for phase in self.monitor:
            state_dict[f"{phase}_monitor"] = {}
            for metric_name, metric in self.monitor[phase].items():
                state_dict[f"{phase}_monitor"][metric_name] = metric.to_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for phase_name in state_dict:
            phase = phase_name.split("_")[0]
            for metric_name in state_dict[phase_name]:
                self.monitor[phase][metric_name].from_dict(state_dict[phase_name][metric_name])