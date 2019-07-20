import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from config.config import Config
from core.engine import Engine
from data.manager import ToyDatasetManager


class MyEngine(Engine):
    def feed_forward_batch(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return {'loss': loss, 'outputs': outputs, 'targets': targets}


if __name__ == "__main__":
    from pathlib import Path
    config = Config()

    datamanager = ToyDatasetManager(config.get('data'))
    train_loader = datamanager.get_loader('training')
    val_loader = datamanager.get_loader('validation')
    test_loader = datamanager.get_loader('test')

    engine = MyEngine(config, datamanager.dim_input, datamanager.dim_target, save_path=Path('ckpts'))
    engine.fit(train_loader, val_loader, num_epochs=50)

    engine = MyEngine(config, datamanager.dim_input, datamanager.dim_target, save_path=Path('ckpts'))
    engine.load(Path('ckpts'), best=False)
    engine.fit(train_loader, val_loader, num_epochs=10)
    engine.evaluate(test_loader)

