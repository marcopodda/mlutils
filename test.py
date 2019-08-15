from pathlib import Path

from mlutils.config import Config
from mlutils.settings import Settings
from mlutils.data.processor import ToyBinaryClassificationDataProcessor
from mlutils.data.provider import DataProvider
from mlutils.core.engine import Engine
from mlutils.experiment import Experiment
from mlutils.experiment.model_selection.selector import ModelSelector


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
    configs = [Config.from_file(f"config{i}.yaml") for i in range(3)]
    settings = Settings('my_settings')
    selector = ModelSelector(
        configs,
        path=Path("MODEL_SELECTION"),
        processor_class=ToyBinaryClassificationDataProcessor,
        provider_class=MyDataProvider)

    print(selector.run())