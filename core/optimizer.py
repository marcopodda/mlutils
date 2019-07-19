import torch

from utils.module_loading import load_class
from .events import EventHandler


class GradientClipper:
    def __init__(self, func, args):
        self.func = func
        self.args = args

    def clip_gradients(self, parameters):
        self.func(parameters, **self.args)


class Optimizer(EventHandler):
    def __init__(self, config, model):
        self.config = config

        opt_config = self.config.get('optimizer')
        self.optimizer = load_class(opt_config, model.parameters())

        self.scheduler = None
        if 'scheduler' in config:
            sched_config = self.config.get('optimizer', 'scheduler')
            self.scheduler = load_class(sched_config, self.optimizer)

        self.gradient_clipper = None
        if 'gradient_clipper' in config:
            grad_clip_config = self.config.get('optimizer', 'gradient_clipper')
            self.gradient_clipper = load_class(grad_clip_config)

    def on_training_epoch_start(self, state):
        if self.scheduler is not None:
            self.scheduler.step()

    def on_training_batch_start(self, state):
        self.optimizer.zero_grad()

    def on_training_batch_end(self, state):
        self.optimizer.step()

    def on_backward(self, state):
        if self.gradient_clipper is not None:
            self.gradient_clipper.clip_gradients(state.model.parameters())

    def state_dict(self):
        state_dict = {'optimizer': self.optimizer.state_dict()}
        if self.scheduler is not None:
            state_dict.update(scheduler=self.scheduler.state_dict())
        return {'optimizer': state_dict}

    def load_state_dict(self, state_dict):
        opt_state_dict = state_dict['optimizer']
        self.optimizer.load_state_dict(opt_state_dict['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(opt_state_dict['scheduler'])