from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
from .events import EventHandler


class GradientClipper(EventHandler):
    def __init__(self, config):
        super().__init__(config)

    def on_parameter_update(self, state):
        if self.config.grad_clip_name == "norm":
            clip = clip_grad_norm_
        elif self.config.grad_clip_name == "value":
            clip = clip_grad_value_

        clip(state.model.parameters(), **self.config.grad_clip_params)