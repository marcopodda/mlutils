class ModelSaver:
    def __init__(self, metric):
        self.metric = metric

    def check_for_save(self, phase, state):
        if phase == self.metric.save_best_on:
            data = self.metric.get_data(phase)
            state.update(save_best=data['is_best'])

    def state_dict(self):
        return self.__dict__.copy()

    def load_state_dict(self, state_dict):
        self.__dict__ = state_dict