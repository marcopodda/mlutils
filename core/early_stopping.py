import operator

from .events import EventHandler


class EarlyStopper(EventHandler):
    def __init__(self, monitor='loss'):
        self.monitor = monitor
        self.best_loss = None


class GLEarlyStopper(EarlyStopper):
    def __init__(self, alpha=5):
        super().__init__()
        self.best_loss = None
        self.alpha = alpha

    def on_validation_epoch_end(self, state):
        val_loss = state.loss.item()

        if val_loss < self.best_loss:
            self.best_loss = val_loss

        stop_condition = 100 * (val_loss / self.best_loss - 1) > self.alpha
        state.update(stop_training=stop_condition)


class PatienceEarlyStopper(EarlyStopper):
    def __init__(self, patience=20):
        super().__init__()
        self.patience = patience
        self.counter = 0

    def on_validation_epoch_end(self, state):
        val_loss = state.loss.item()

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        stop_condition = self.counter > self.patience
        state.update(stop_training=stop_condition)