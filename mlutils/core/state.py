import torch
import numpy as np

from utils.training import pretty_print
from .loggers import logger


class State:
    def __init__(self, **values):
        self.update(**values)
        self.epoch = 0
        self.stop_training = False
        self.epoch_results = {}
        self.best_results = {}
        self.results = []

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return name in self.__dict__

    def update(self, **values):
        for name, value in values.items():
            setattr(self, name, value)

    def init_epoch_results(self, metrics):
        for metric in metrics:
            key = f"{self.phase}_{metric.name}"

            if key not in self.epoch_results:
                self.epoch_results.update(**{key: {}})

            if key not in self.best_results:
                best = -np.float('inf') if metric.greater_is_better else np.float('inf')
                self.best_results.update(**{key: {'best': best, 'best_epoch': 0}})

    def update_epoch_results(self, metrics):
        for metric in metrics:
            key = f"{self.phase}_{metric.name}"
            self.epoch_results.update(**metrics.get_data(self.phase))
            best_value = self.best_results[key]['best']
            current_value = metric.get_value(self.phase)
            if metric.operator(current_value, best_value):
                self.best_results[key]['best'] = current_value
                self.best_results[key]['best_epoch'] = self.epoch

    def save_epoch_results(self):
        self.results.append(self.epoch_results)
        logger.info(pretty_print(self.epoch_results))

    def state_dict(self):
        state_dict = self.__dict__.copy()
        for key, obj in state_dict.items():
            if hasattr(obj, 'state_dict'):
                state_dict[key] = obj.state_dict()
        return {'state': state_dict}

    def load(self, filename):
        logger.info(f"Loading state file {filename}")
        state_dict = torch.load(filename)
        self.model.load_state_dict(state_dict['model_state'])
        self.criterion.load_state_dict(state_dict['criterion_state'])
        self.optimizer_state = state_dict['optimizer_state']
        self.scheduler_state = state_dict['scheduler_state']
        self.epoch = state_dict['epoch'] + 1
        self.best_results = state_dict['best_results']

    def save(self, filename):
        state_dict = {
            'epoch': self.epoch,
            'best_results': self.best_results,
            'model_state': self.model.state_dict(),
            'criterion_state': self.criterion.state_dict(),
            'optimizer_state': self.optimizer_state,
            'scheduler_state': self.scheduler_state
        }
        logger.info(f"Saving state to file {filename}")
        torch.save(state_dict, filename)