import torch


class EventDispatcher:
    def __init__(self):
        self._event_handlers = []

    def _dispatch(self, event_name, state):
        for event_handler in self._event_handlers:
            callback = getattr(event_handler, event_name)
            callback(state)

    def register(self, event_handler):
        self._event_handlers.append(event_handler)

    def unregister(self, event_handler):
        self._event_handlers.remove(event_handler)

    def register_all(self):
        for event_handler in self._event_handlers:
            self.register(event_handler)

    def unregister_all(self):
        for event_handler in self._event_handlers:
            self.unregister(event_handler)


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

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class State:
    def __init__(self, **values):
        self.update(**values)

    def update(self, **values):
        for name, value in values.items():
            setattr(self, name, value)

    def remove(self, *names):
        for name in names:
            delattr(self, name)

    def state_dict(self):
        state_dict = self.__dict__.copy()
        for key, obj in state_dict.items():
            if hasattr(obj, 'state_dict'):
                state_dict[key] = obj.state_dict()
        return state_dict

    def load(self, path):
        state_dict = torch.load(path)
        for key, obj in state_dict.items():
            if hasattr(self, 'load_state_dict'):
                self.update(key=obj.load_state_dict())
            else:
                self.update(key=obj)
