from mlutils.settings import Settings
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
    settings = Settings('my_settings')
    exp = Experiment("config.yaml")
    exp.run()
