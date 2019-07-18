import torch

from utils.module_loading import import_string
from .events import EventHandler


class Optimizer(EventHandler):
    def __init__(self, config, model):
        opt_class = import_string(config.class_name)
        self.optimizer = opt_class(model.parameters(), **config.params)

        self.scheduler = None
        sched_config = config.get('scheduler')
        if sched_config != {}:
            sched_class = import_string(sched_config.class_name)
            self.scheduler = sched_class(self.optimizer, **sched_config.params)

        self.grad_clipper = None
        grad_clipper_config = config.get('gradient_clipper')
        if grad_clipper_config != {}:
            self.grad_clipper = import_string(grad_clipper_config.func)
            self.grad_clipper_args = grad_clipper_config.args


    def on_training_epoch_start(self, state):
        if self.scheduler is not None:
            self.scheduler.step()

    def on_training_batch_start(self, state):
        self.optimizer.zero_grad()

    def on_training_batch_end(self, state):
        self.optimizer.step()

    def on_backward(self, state):
        if self.grad_clipper is not None:
            self.grad_clipper(state.model.parameters(), **self.grad_clipper_args)

    def state_dict(self):
        state_dict = {'optimizer': self.optimizer.state_dict()}
        if self.scheduler is not None:
            state_dict.update(scheduler=self.scheduler.state_dict())
        return state_dict

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler'])