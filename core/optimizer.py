from utils.module_loading import import_string
from .events import EventHandler


class Optimizer(EventHandler):
    def __init__(self, config):
        self.optimizer_class = import_string(config.optimizer_class)
        self.optimizer_params = config.optimizer_params

        if config.scheduler_class:
            self.scheduler_class = import_string(config.scheduler_class)
            self.scheduler_params = config.scheduler_params

    def on_fit_start(self, state):
        self.optimizer = self.optimizer_class(state.model.parameters(), **self.optimizer_params)
        self.scheduler = self.scheduler_class(self.optimizer, **self.scheduler_params)

    def on_training_epoch_start(self, state):
        if self.scheduler:
            self.scheduler.step()

    def on_training_batch_start(self, state):
        self.optimizer.zero_grad()

    def on_training_batch_end(self, state):
        self.optimizer.step()