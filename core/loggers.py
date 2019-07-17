import torch

from .events import EventHandler
from utils.module_loading import import_string


class LossLogger(EventHandler):
    def __init__(self, path=None):
        self.losses = [] if path is None else torch.load(path / 'losses.pt')
        self.batch_losses = []

    def on_training_batch_start(self, state):
        self.batch_losses = []

    def on_training_batch_end(self, state):
        self.batch_losses.append(state.loss.item())

    def on_training_epoch_end(self, state):
        epoch_loss = sum(self.batch_losses) / len(self.batch_losses)
        self.losses.append(epoch_loss)
        print(f"training loss: {self.losses[-1]:.6f}")

    def save(self, path):
        torch.save(self.losses, path / 'losses.pt')

    def on_fit_end(self, state):
        print(len(self.losses), self)



class MetricLogger(EventHandler):
    def __init__(self, metrics=[], path=None):
        self.metrics = []

        for class_name in metrics:
            metric_class = import_string(class_name)
            metric = metric_class(path=path, phase=self.phase)
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

    def save(self, path):
        for metric in self.metrics:
            torch.save(metric.values, path / f"{self.phase}_{metric.name}.pt")


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