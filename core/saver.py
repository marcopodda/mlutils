from pathlib import Path

from .events import EventHandler

class ModelSaver(EventHandler):
    def __init__(self, path=Path('ckpts'), monitor='validation_loss'):
        self.path = path
        self.monitor = monitor

    def on_fit_end(self, state):
        print(f'Saving last at epoch {state.epoch}')
        state.save(self.path / "last.pt")

    def on_epoch_end(self, state):
        if state.best_results[self.monitor]['best_epoch'] == state.epoch:
            print(f'Saving best at epoch {state.epoch}')
            state.save(self.path / "best.pt")
