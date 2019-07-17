import torch

from utils.module_loading import import_string
from .events import EventHandler


class Optimizer(EventHandler):
    def __init__(self, config, model):
        opt_config = config.get('optimizer')
        opt_class = import_string(opt_config.class_name)
        self.optimizer = opt_class(model.parameters(), **opt_config.params)

        self.scheduler = None
        sched_config = config.get('optimizer', 'scheduler')
        if sched_config != {}:
            sched_class = import_string(sched_config.class_name)
            self.scheduler = sched_class(self.optimizer, **sched_config.params)

    def on_training_epoch_start(self, state):
        if self.scheduler:
            self.scheduler.step()

    def on_training_batch_start(self, state):
        self.optimizer.zero_grad()

    def on_training_batch_end(self, state):
        self.optimizer.step()

    def state_dict(self):
        state_dict = {'optimizer': self.optimizer.state_dict()}
        if self.scheduler:
            state_dict.update(scheduler=self.scheduler.state_dict())
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(state_dict['scheduler'])