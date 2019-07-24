import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from mlutils.config import Config
from mlutils.data.processor import ToyBinaryClassificationDataProcessor
from mlutils.data.provider import DataProvider
from mlutils.core.engine import Engine
from mlutils.experiment import Experiment


class MyDataProvider(DataProvider):
    @property
    def dim_features(self):
        return 16

    @property
    def dim_target(self):
        return 1


class MyEngine(Engine):
    def feed_forward_batch(self, batch):
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        return {'loss': loss, 'outputs': outputs, 'targets': targets}


if __name__ == "__main__":
    exp = Experiment("config.yaml",
                     processor_class=ToyBinaryClassificationDataProcessor,
                     provider_class=MyDataProvider,
                     engine_class=MyEngine)
    exp.run()

