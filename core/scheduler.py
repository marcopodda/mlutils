from torch import optim
from torch.optim import lr_scheduler

from .events import EventHandler


class Scheduler(EventHandler):
    def __init__(self, config):
        super().__init__(config)
        self.scheduler_class = getattr(lr_scheduler, config.scheduler_name)
        self.scheduler = None

    def on_fit_start(self, state):
        self.scheduler = self.scheduler_class(state.optimizer, **self.config.scheduler_params)

    def on_epoch_start(self, state):
        self.scheduler.step()