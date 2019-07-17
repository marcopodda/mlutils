import torch

from .events import EventHandler
from utils.module_loading import import_string


class LossLogger(EventHandler):
    def __init__(self):
        self.losses = []
        self.batch_losses = []

    def _on_batch_start(self, state):
        self.batch_losses = []

    def _on_batch_end(self, state):
        self.batch_losses.append(state.loss.item())

    def _on_epoch_end(self, state):
        epoch_loss = sum(self.batch_losses) / len(self.batch_losses)
        self.losses.append(epoch_loss)
        print(f"{self.phase} loss: {self.losses[-1]:.6f}")

    def state_dict(self):
        return {f"{self.phase}_losses": self.losses}

    def load_state_dict(self, state_dict):
        if 'losses' in state_dict:
            self.losses = state_dict[f"{self.phase}_losses"]


class TrainingLossLogger(LossLogger):
    phase = "training"

    def on_training_batch_start(self, state):
        self._on_batch_start(state)

    def on_training_batch_end(self, state):
        self._on_batch_end(state)

    def on_training_epoch_end(self, state):
        self._on_epoch_end(state)


class ValidationLossLogger(LossLogger):
    phase = "validation"

    def on_validation_batch_start(self, state):
        self._on_batch_start(state)

    def on_validation_batch_end(self, state):
        self._on_batch_end(state)

    def on_validation_epoch_end(self, state):
        self._on_epoch_end(state)


class TestLossLogger(LossLogger):
    phase = "test"

    def on_test_batch_start(self, state):
        self._on_batch_start(state)

    def on_test_batch_end(self, state):
        self._on_batch_end(state)

    def on_test_end(self, state):
        self._on_epoch_end(state)


class MetricLogger(EventHandler):
    def __init__(self, metrics=[]):
        self.metrics = []

        for class_name in metrics:
            metric_class = import_string(class_name)
            metric = metric_class(phase=self.phase)
            self.metrics.append(metric)

        self.outputs = []
        self.targets = []

    def _on_epoch_start(self, state):
        self.outputs = []
        self.targets = []

    def _on_batch_end(self, state):
        self.outputs.append(state.outputs.detach())
        self.targets.append(state.targets.detach())

    def _on_epoch_end(self, state):
        for i, metric in enumerate(self.metrics):
            outputs = torch.cat(self.outputs)
            targets = torch.cat(self.targets)
            metric.update(outputs, targets)
            print(f"{self.phase} {metric.name}: {metric.value():.6f}")

    def state_dict(self):
        state_dict = {f'{self.phase}_metrics': {}}
        for metric in self.metrics:
            state_dict[f'{self.phase}_metrics'].update({metric.name: metric.values})
        return state_dict

    def load_state_dict(self, state_dict):
        if f'{self.phase}_metrics' in state_dict:
            for metric in self.metrics:
                if metric.name in state_dict[f'{self.phase}_metrics']:
                    metric.values = state_dict[f'{self.phase}_metrics'][metric.name]


class TrainingMetricLogger(MetricLogger):
    phase = "training"

    def on_training_epoch_start(self, state):
        self._on_epoch_start(state)

    def on_training_batch_end(self, state):
        self._on_batch_end(state)

    def on_training_epoch_end(self, state):
        self._on_epoch_end(state)


class ValidationMetricLogger(MetricLogger):
    phase = "validation"

    def on_validation_epoch_start(self, state):
        self._on_epoch_start(state)

    def on_validation_batch_end(self, state):
        self._on_batch_end(state)

    def on_validation_epoch_end(self, state):
        self._on_epoch_end(state)


class TestMetricLogger(MetricLogger):
    phase = "test"

    def on_test_start(self, state):
        self._on_epoch_start(state)

    def on_test_batch_end(self, state):
        self._on_batch_end(state)

    def on_test_end(self, state):
        self._on_epoch_end(state)