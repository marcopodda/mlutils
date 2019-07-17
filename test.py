import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from config.config import Config
from core.engine import Engine
from core.metrics import BinaryAccuracy
from core.loggers import TrainingMetricLogger, ValidationMetricLogger, LossLogger
from core.gradient_clipping import GradientClipper
from data.manager import DataManager
from data.datasets import RandomDataset
from sklearn.datasets import make_classification


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
    from pathlib import Path
    config = Config()

    datamanager = Manager(config)
    train_loader = datamanager.get_loader('training')
    val_loader = datamanager.get_loader('validation')

    engine = MyEngine(config, datamanager.dim_input, datamanager.dim_target) #, event_handlers=event_handlers)
    engine.fit(train_loader, val_loader, max_epochs=2)
    print(engine._event_handlers)
    engine.save(Path('ckpts'))
    engine.unregister_all()
    # del engine
    datamanager = Manager(config)
    train_loader = datamanager.get_loader('training')
    val_loader = datamanager.get_loader('validation')
    engine2 = MyEngine(config, datamanager.dim_input, datamanager.dim_target, path=Path('ckpts'))

    engine2.fit(train_loader, val_loader, max_epochs=3)
    print(engine2._event_handlers)


