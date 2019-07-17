import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data):
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


class RandomDataset(Dataset):
    def __getitem__(self, index):
        inputs = self._data[index, :-1]
        targets = self._data[index, -1].reshape(-1, 1)
        return torch.FloatTensor(inputs), torch.FloatTensor(targets)

    @property
    def targets(self):
        return self._data[:,-1]

    @property
    def dim_input(self):
        return self._data[:,:-1].shape[1]

    @property
    def dim_target(self):
        return 1