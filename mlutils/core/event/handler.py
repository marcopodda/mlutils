class EventHandler:
    def on_fit_start(self, state):
        pass

    def on_fit_end(self, state):
        pass

    def on_epoch_start(self, state):
        pass

    def on_epoch_end(self, state):
        pass

    def on_training_epoch_start(self, state):
        pass

    def on_training_epoch_end(self, state):
        pass

    def on_validation_epoch_start(self, state):
        pass

    def on_validation_epoch_end(self, state):
        pass

    def on_training_batch_start(self, state):
        pass

    def on_training_batch_end(self, state):
        pass

    def on_validation_batch_start(self, state):
        pass

    def on_validation_batch_end(self, state):
        pass

    def on_backward(self, state):
        pass

    def on_test_epoch_start(self, state):
        pass

    def on_test_epoch_end(self, state):
        pass

    def on_test_batch_start(self, state):
        pass

    def on_test_batch_end(self, state):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
