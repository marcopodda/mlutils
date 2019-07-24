from mlutils.core.event.handler import EventHandler
from mlutils.util.module_loading import load_class, import_string


class GradientClipper:
    def __init__(self, func, args):
        self.func = import_string(func)
        self.args = args

    def clip_gradients(self, parameters):
        self.func(parameters, **self.args)


class Optimizer(EventHandler):
    def __init__(self, config, model):
        self.optimizer = load_class(config, model.parameters())
        self.model = model

        self.scheduler = None
        if 'scheduler' in config:
            self.scheduler = load_class(config.scheduler, optimizer=self.optimizer)

        self.gradient_clipper = None
        if 'gradient_clipper' in config:
            self.gradient_clipper = load_class(config.gradient_clipper)

    def on_fit_start(self, state):
        if 'optimizer_state' in state:
            self.optimizer.load_state_dict(state.optimizer_state)

        if self.scheduler and 'scheduler_state' in state:
            self.scheduler.load_state_dict(state.scheduler_state)

    def on_training_epoch_start(self, state):
        if self.scheduler is not None:
            self.scheduler.step()

    def on_training_batch_start(self, state):
        self.optimizer.zero_grad()

    def on_training_batch_end(self, state):
        self.optimizer.step()

    def on_backward(self, state):
        if self.gradient_clipper is not None:
            self.gradient_clipper.clip_gradients(self.model.parameters())

    def on_epoch_end(self, state):
        state.update(optimizer_state=self.optimizer.state_dict())
        if self.scheduler:
            state.update(scheduler_state=self.scheduler.state_dict())
