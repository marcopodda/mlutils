from .events import EventHandler


class Saver(EventHandler):
    def __init__(self, config):
        self.phase = config.phase
        self.metric = config.metric

    def _on_epoch_end(self, phase, state):
        print(state.__dict__)
        if phase == self.phase:
            is_best = state[f"{self.phase}_{self.metric}"]['is_best']
            state.update(save_best=is_best is True)


    def on_validation_epoch_end(self, state):
        return self._on_epoch_end('validation', state)

    def on_training_epoch_end(self, state):
        return self._on_epoch_end('training', state)