import torch
from torch.utils.data import Dataset


class FileDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError

    @property
    def dim_features(self):
        raise NotImplementedError

    @property
    def dim_target(self):
        raise NotImplementedError
