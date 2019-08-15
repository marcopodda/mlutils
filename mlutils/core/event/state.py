import torch
import numpy as np

from mlutils.util.training import pretty_print
from mlutils.core.logging import Logger


class State:
    def __init__(self, **values):
        self.update(**values)
        self.epoch = 0
        self.stop_training = False
        self.epoch_results = {}
        self.best_results = {}
        self.results = []
        self.logger = Logger()

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return name in self.__dict__

    def update(self, **values):
        for name, value in values.items():
            setattr(self, name, value)

    def save_epoch_results(self):
        self.results.append(self.epoch_results)
        self.logger.success(pretty_print(self.epoch_results))

    def load(self, filename):
        self.logger.info(f"Loading state file {filename}")
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict['model_state'])
        self.criterion.load_state_dict(state_dict['criterion_state'])
        self.optimizer_state = state_dict['optimizer_state']
        if 'scheduler_state' in state_dict:
            self.scheduler_state = state_dict['scheduler_state']
        self.epoch = state_dict['epoch'] + 1
        self.best_results = state_dict['best_results']

    def save(self, filename):
        state_dict = {
            'epoch': self.epoch,
            'best_results': self.best_results,
            'model_state': self.model.state_dict(),
            'criterion_state': self.criterion.state_dict(),
            'optimizer_state': self.optimizer_state
        }
        if 'scheduler_state' in self:
            state_dict.update({
                'scheduler_state': self.scheduler_state
            })
        self.logger.info(f"Saving state to file {filename}")
        torch.save(state_dict, filename)
