from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
from .events import EventHandler


class GradientClipper(EventHandler):
    def __init__(self, config):
        self.clip_value = config.grad_clip_params['clip_value']

    def on_backward(self, state):
        clip_grad_value_(state.model.parameters(), clip_value=self.clip_value)


class GradientNormClipper(EventHandler):
    def __init__(self, config):
        self.max_norm = config.grad_clip_params['max_norm']
        self.norm_type = config.grad_clip_params.get('norm_type', 2)

    def on_backward(self, state):
        clip_grad_norm_(state.model.parameters(), max_norm=self.max_norm, norm_type=self.norm_type)