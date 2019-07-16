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

    def on_epoch_end(self, state):
        print("training loss", self.losses[-1], end=" - ")



class MetricLogger(EventHandler):
    def __init__(self, metrics):
        self.metrics = metrics
        self.outputs = []
        self.targets = []

    def on_training_epoch_start(self, state):
        self.outputs = []
        self.targets = []

    def on_training_batch_end(self, state):
        self.outputs.append(state.outputs.detach())
        self.targets.append(state.targets.detach())

    def on_training_epoch_end(self, state):
        for name, metric in self.metrics.items():
            if "train" in name:
                outputs = torch.cat(self.outputs)
                targets = torch.cat(self.targets)
                metric.update(outputs, targets)

    def on_validation_start(self, state):
        self.outputs = []
        self.targets = []

    def on_validation_batch_end(self, state):
        self.outputs.append(state.outputs.detach())
        self.targets.append(state.targets.detach())

    def on_validation_epoch_end(self, state):
        for name, metric in self.metrics.items():
            if "val" in name:
                outputs = torch.cat(self.outputs)
                targets = torch.cat(self.targets)
                metric.update(outputs, targets)

    def on_epoch_end(self, state):
        print("training accuracy", self.metrics['train_acc'].value(), end=" - ")
        print("validation accuracy", self.metrics['val_acc'].value())