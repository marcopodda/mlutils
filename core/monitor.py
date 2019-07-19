import numpy as np

from utils.module_loading import dynamic_class_load
from .events import EventHandler
from .loggers import CSVLogger


class Monitor(EventHandler):
    def __init__(self, config):
        self.metrics = []
        self.early_stopper = None
        self.saver = None
        self.logger = None

        monitor_config = config.get('monitor')
        if 'metrics' in monitor_config:
            metric_configs = monitor_config.get('metrics')
            for metric_config in metric_configs:
                metric = dynamic_class_load(metric_config)
                self.metrics.append(metric)

            if 'early_stopper' in monitor_config:
                metric = self.get_early_stopper_metric()
                early_stopper_config = monitor_config.get('early_stopper')
                self.early_stopper = dynamic_class_load(early_stopper_config, metric)

            if 'saver' in monitor_config:
                metric = self.get_save_best_metric()
                saver_config = monitor_config.get('saver')
                self.saver = dynamic_class_load(saver_config, metric)

            if 'logger' in monitor_config:
                logger_config = monitor_config.get('logger')
                self.logger = CSVLogger(**logger_config.params)
                self.logger.init(self.metrics)

    def get_early_stopper_metric(self):
        return self.metrics[0]

    def get_save_best_metric(self):
        return self.metrics[0]

    def _on_epoch_end(self, phase, state):
        for metric in self.metrics:
            metric.update(phase)

        if self.early_stopper is not None:
            self.early_stopper.check_for_early_stop(phase, state)

        if self.saver is not None:
            self.saver.check_for_save(phase, state)

        if self.logger is not None:
            values = {}
            for metric in self.metrics:
                try:
                    values.update({metric.name: metric.data[phase]['current']})
                except KeyError:
                    values.update({metric.name: None})
            self.logger.log(phase, values)

    def _on_batch_start(self):
        for metric in self.metrics:
            metric.reset_batch_data()

    def _on_batch_end(self, state):
        for metric in self.metrics:
            metric.update_batch_data(state)

    def on_training_batch_end(self, state):
        self._on_batch_end(state)

    def on_training_batch_start(self, state):
        self._on_batch_start()

    def on_training_epoch_end(self, state):
        self._on_epoch_end('training', state)

    def on_validation_batch_end(self, state):
        self._on_batch_end(state)

    def on_validation_batch_start(self, state):
        self._on_batch_start()

    def on_validation_epoch_end(self, state):
        self._on_epoch_end('validation', state)

    def on_test_batch_end(self, state):
        self._on_batch_end(state)

    def on_test_batch_start(self, state):
        self._on_batch_start()

    def on_test_end(self, state):
        self._on_epoch_end('test', state)
        self.logger._empty_buffer('test')

    def on_fit_end(self, state):
        for phase in ['training', 'validation']:
            self.logger._empty_buffer(phase)

    def state_dict(self):
        state_dict = {'metrics': [m.state_dict() for m in self.metrics]}
        if self.early_stopper is not None:
            state_dict.update(early_stopper=self.early_stopper.state_dict())
        if self.saver is not None:
            state_dict.update(saver=self.saver.state_dict())
        return {'monitor': state_dict}

    def load_state_dict(self, state_dict):
        monitor_state_dict = state_dict['monitor']
        for metric, metric_state_dict in zip(self.metrics, monitor_state_dict['metrics']):
            metric.load_state_dict(metric_state_dict)

        if 'early_stopper' in monitor_state_dict:
            self.early_stopper.load_state_dict(monitor_state_dict['early_stopper'])

        if 'saver' in monitor_state_dict:
            self.saver.load_state_dict(monitor_state_dict['saver'])







