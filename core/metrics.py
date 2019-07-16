import numpy as np

import torch
from sklearn import metrics
from .events import EventHandler


class Metric(EventHandler):
    def __init__(self, config):
        super().__init__(config)
        self.outputs = []
        self.targets = []
        self.values = []

    def on_batch_end(self, state):
        if state.training is True:
            self.outputs.extend(state.outputs.detach())
            self.targets.extend(state.targets.detach())

    def on_epoch_start(self, state):
        self.outputs = []
        self.targets = []

    def on_epoch_end(self, state):
        if state.training is True:
            metric = self.get_metric()
            outputs, targets = self._prepare_data()
            value = metric(targets, outputs)
            print("accuracy", value)
            self.values.append(value)

    def _prepare_data(self):
        raise NotImplementedError

    def get_metric(self):
        raise NotImplementedError


class Loss(EventHandler):
    def __init__(self, config):
        super().__init__(config)
        self.loss = 0

    def on_batch_end(self, state):
        self.loss += state.loss

    def on_epoch_end(self, state):
        if state.training is True:
            print("loss", self.loss / state.num_batches)

    def on_epoch_start(self, state):
        if state.training is True:
            self.loss = 0



class Accuracy(Metric):
    def get_metric(self):
        return metrics.accuracy_score

    def _prepare_data(self):
        outputs = torch.sigmoid(torch.cat(self.outputs))
        targets = torch.cat(self.targets)
        return (outputs > 0.5).numpy(), (targets == 1).numpy()


class MSE(Metric):
    function = metrics.mean_squared_error

    def _prepare_data(self):
        return self.outputs, self.targets