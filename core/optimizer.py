import torch

from utils.module_loading import import_string
from .events import EventHandler


class Optimizer(EventHandler):
    def __init__(self, config, model, path=None):
        opt_config = config.get('optimizer')
        opt_class = import_string(opt_config.class_name)
        self.optimizer = opt_class(model.parameters(), **opt_config.params)

        if path is not None:
            self.optimizer.load_state_dict(torch.load(path / 'optimizer.pt'))

        self.scheduler = None
        sched_config = config.get('optimizer', 'scheduler')
        if sched_config != {}:
            sched_class = import_string(sched_config.class_name)
            self.scheduler = sched_class(self.optimizer, **sched_config.params)
            if path is not None:
                self.scheduler.load_state_dict(torch.load(path / 'scheduler.pt'))

    def on_training_epoch_start(self, state):
        if self.scheduler:
            self.scheduler.step()

    def on_training_batch_start(self, state):
        self.optimizer.zero_grad()

    def on_training_batch_end(self, state):
        self.optimizer.step()

    def save(self, path):
        torch.save(self.optimizer.state_dict(), path / 'optimizer.pt')

        if self.scheduler:
            torch.save(self.scheduler.state_dict(), path / 'scheduler.pt')