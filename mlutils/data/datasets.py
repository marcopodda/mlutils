import numpy as np

import torch
from torch.utils import data

from sklearn.datasets import make_classification, make_regression
from sklearn.utils.multiclass import unique_labels


class Dataset(data.Dataset):
    def __init__(self, data=[]):
        self._data = data

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self._data)

    @property
    def indices(self):
        return list(range(len(self._data)))

    @property
    def targets(self):
        raise NotImplementedError

    @property
    def dim_input(self):
        raise NotImplementedError

    @property
    def dim_target(self):
        raise NotImplementedError


class Collator:
    def __init__(self, config):
        self.config = config

    def __call__(self, batch):
        raise NotImplementedError


class ToyClassificationDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(data=[])

        X, Y = make_classification(**kwargs)
        self._data = np.hstack([X, Y.reshape(-1, 1)])
        self.num_features = kwargs['n_features']

    def __getitem__(self, index):
        inputs = torch.FloatTensor(self._data[index, :-1])
        targets = self._data[index, -1].reshape(-1, 1)

        if self.dim_target <= 2:
            targets = torch.FloatTensor(targets)
        else:
            targets = torch.LongTensor(targets)

        return inputs, targets

    @property
    def targets(self):
        return self._data[:,-1]

    @property
    def dim_input(self):
        return self.num_features

    @property
    def dim_target(self):
        num_targets = len(unique_labels(self.targets))
        return 1 if num_targets <= 2 else num_targets


class ToyRegressionDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(data=[])

        X, Y = make_regression(**kwargs)
        self._data = np.hstack([X, Y.reshape(-1, 1)])
        self.num_features = kwargs['n_features']

    def __getitem__(self, index):
        inputs = torch.FloatTensor(self._data[index, :-1])
        targets = self._data[index, -1].reshape(-1, 1)

        if self.dim_target <= 2:
            targets = torch.FloatTensor(targets)
        else:
            targets = torch.LongTensor(targets)

        return inputs, targets

    @property
    def targets(self):
        return self._data[:,-1]

    @property
    def dim_input(self):
        return self.num_features

    @property
    def dim_target(self):
        return 1


