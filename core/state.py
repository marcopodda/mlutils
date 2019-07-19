class State:
    def __init__(self, **values):
        self.update(**values)
        self.epoch_results = {}
        self.results = []

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return name in self.__dict__

    def update(self, **values):
        for name, value in values.items():
            setattr(self, name, value)

    def init_epoch_results(self):
        self.epoch_results = {}

    def update_epoch_results(self, **values):
        self.epoch_results.update(**values)

    def save_epoch_results(self):
        self.results.append(self.epoch_results)

    def remove(self, *names):
        for name in names:
            delattr(self, name)

    def state_dict(self):
        state_dict = self.__dict__.copy()
        for key, obj in state_dict.items():
            if hasattr(obj, 'state_dict'):
                state_dict[key] = obj.state_dict()
        return {'state': state_dict}

    def load_state_dict(self, state_dict):
        state_dict = state_dict['state']
        for key, state in state_dict.items():
            if state is not None and key in self and hasattr(self[key], 'load_state_dict'):
                self[key].load_state_dict(state)
            else:
                self.update(**{key: state})