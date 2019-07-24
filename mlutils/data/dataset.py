import torch
from torch.utils.data import Dataset


class FileDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError


class ToyBinaryClassificationDataset(FileDataset):
    def __getitem__(self, index):
        data = self.data[index]
        inputs = torch.FloatTensor(data[:-1])
        target = torch.FloatTensor(data[-1:])
        return inputs, target