import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from config.config import Config
from core.engine import Engine
from core.metrics import BinaryAccuracy
from core.loggers import TrainingMetricLogger, ValidationMetricLogger, LossLogger
from core.optimizer import Optimizer
from core.gradient_clipping import GradientNormClipper
from data.manager import DataManager
from data.datasets import Dataset
from sklearn.datasets import make_classification



class Model(nn.Module):
    def __init__(self, config, dim_input, dim_target):
        super().__init__()
        self.config = config
        self.dim_input = dim_input
        self.dim_target = dim_target

        self.linear1 = nn.Linear(dim_input, config.dim_hidden)
        self.linear2 = nn.Linear(config.dim_hidden, config.dim_hidden)
        self.linear3 = nn.Linear(config.dim_hidden, dim_target)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class Criterion(nn.Module):
    def __init__(self, config, dim_target):
        super().__init__()
        self.config = config
        self.dim_target = dim_target

        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        return self.bce(outputs, targets.squeeze(-1))


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


class Manager(DataManager):
    def _fetch_data(self, raw_dir):
        pass

    def _process_data(self, processed_dir):
        X, Y = make_classification(n_samples=10000, n_features=100, n_informative=2, n_redundant=2, n_repeated=2, n_classes=2)

        data = RandomDataset(np.hstack([X, Y.reshape(-1, 1)]))
        torch.save(data, processed_dir / "dataset.pt")


class MyEngine(Engine):
    def process_batch(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return {'loss': loss, 'outputs': outputs, 'targets': targets}


if __name__ == "__main__":
    config_dict = {
        'dim_hidden': 128,
        'optimizer_class': 'torch.optim.Adam',
        'optimizer_params': {
            'lr': 0.001
        },
        'data_root': "DATA",
        'dataset_name': 'random',
        'splitter_class': 'data.splitters.HoldoutSplitter',
        'splitter_params': {},
        'dataloader_params': {
            'batch_size': 32,
            'shuffle': True
        },
        'max_epochs': 1000,
        'scheduler_class': 'torch.optim.lr_scheduler.StepLR',
        'scheduler_params': {
            'gamma': 0.5,
            'step_size': 30
        },
        'grad_clip_name': 'norm',
        'grad_clip_params': {
            'max_norm': 1.0,
            'norm_type': 2
        }
    }

    config = Config(**config_dict)
    train_metrics = [BinaryAccuracy()]
    val_metrics = [BinaryAccuracy()]

    event_handlers = [
        Optimizer(config),
        LossLogger(),
        GradientNormClipper(config),
        TrainingMetricLogger(train_metrics),
        ValidationMetricLogger(val_metrics)]

    datamanager = Manager(config)
    train_loader = datamanager.get_loader('training')
    val_loader = datamanager.get_loader('validation')

    engine = MyEngine(config, Model, Criterion, datamanager.dim_input, datamanager.dim_target, event_handlers=event_handlers)
    engine.fit(train_loader, val_loader)

