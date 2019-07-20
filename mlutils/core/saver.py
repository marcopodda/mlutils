from pathlib import Path

from .events import EventHandler
from .loggers import logger

class ModelSaver(EventHandler):
    def __init__(self, path=Path('ckpts'), monitor='validation_loss'):
        self.path = path
        self.monitor = monitor

    def on_fit_end(self, state):
        filename = self.path / "last.pt"
        logger.info(f"Saving last model in {filename}")
        state.save(filename)

    def on_epoch_end(self, state):
        if state.best_results[self.monitor]['best_epoch'] == state.epoch:
            filename = self.path / "best.pt"
            logger.info(f"Found new best model at epoch {state.epoch}. Saving in {filename}")
            state.save(filename)
