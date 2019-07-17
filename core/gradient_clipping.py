from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_

from utils.module_loading import import_string
from .events import EventHandler


class GradientClipper(EventHandler):
    def __init__(self, func=None, args=None, **kwargs):
        self.func = func
        self.func_args = args

    def on_backward(self, state):
        clip_grad_func = import_string(self.func)
        clip_grad_func(state.model.parameters(), **self.func_args)