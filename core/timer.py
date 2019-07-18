import time

from .events import EventHandler

class Timer(EventHandler):
    def __init__(self):
        self.epoch_times = []
        self.training_epoch_times = []
        self.validation_epoch_times = []
        self.test_time = 0

        self.start_training_epoch = None
        self.start_validation_epoch = None
        self.start_test = None
        self.total_training_time = None
        self.total_validation_time = None
        self.total_fit_time = None

    def on_training_epoch_start(self, state):
        self.start_training_epoch = time.time()

    def on_validation_epoch_start(self, state):
        self.start_validation_epoch = time.time()

    def on_test_start(self, state):
        self.start_test = time.time()

    def on_training_epoch_end(self, state):
        self.training_epoch_times.append(time.time() - self.start_training_epoch)

    def on_validation_epoch_end(self, state):
        self.validation_epoch_times.append(time.time() - self.start_validation_epoch)

    def on_test_end(self, state):
        self.test_time = time.time() - self.start_test

    def on_epoch_end(self, state):
        total_epoch_time = self.training_epoch_times[-1] + self.validation_epoch_times[-1]
        self.epoch_times.append(total_epoch_time)

    def on_fit_end(self, state):
        self.total_training_time = sum(self.training_epoch_times)
        self.total_validation_time = sum(self.validation_epoch_times)
        self.total_fit_time = sum(self.epoch_times)

    def state_dict(self):
        return {'timer': self.__dict__.copy()}

    def load_state_dict(self, state_dict):
        self.__dict__ = state_dict['timer']