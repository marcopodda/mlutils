import torch

from .events import EventHandler


class LossLogger(EventHandler):
    def __init__(self):
        self.losses = []
        self.batch_losses = []

    def on_training_batch_start(self, state):
        self.batch_losses = []

    def on_training_batch_end(self, state):
        self.batch_losses.append(state.loss.item())

    def on_backward(self, state):
        state.loss.backward()

    def on_training_epoch_end(self, state):
        epoch_loss = sum(self.batch_losses) / len(self.batch_losses)
        self.losses.append(epoch_loss)
        print(f"training loss: {self.losses[-1]:.6f}")



class MetricLogger(EventHandler):
    def __init__(self, metrics):
        self.metrics = metrics
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