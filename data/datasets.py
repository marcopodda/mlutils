import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, inputs, targets=None, indices=None):
        self._inputs = inputs
        self._targets = targets
        self._indices = indices

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self._inputs)

    @property
    def targets(self):
        return self._targets

    @property
    def indices(self):
        if self._indices is not None:
            return self._indices
        return list(range(len(self._inputs)))

    @property
    def dim_input(self):
        raise NotImplementedError

    @property
    def dim_output(self):
        raise NotImplementedError


class Collator:
    def __init__(self, config):
        self.config = config

    def __call__(self, batch):
        raise NotImplementedError


def get_dataloader(dataset, config):
    dataloader = data.DataLoader(dataset, batch_size=config.batch_size)
    return dataloader

