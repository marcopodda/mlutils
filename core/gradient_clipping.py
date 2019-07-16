from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_

from utils.module_loading import import_string
from .events import EventHandler


class GradientClipper(EventHandler):
    def __init__(self, config):
        self.config = config

    def on_backward(self, state):
        clip_grad_func = import_string(self.config.grad_clip_func)
        clip_grad_func(state.model.parameters(), **self.config.grad_clip_params)