from pathlib import Path

from mlutils.config import Config
from mlutils.settings import Settings
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
    config = Config.from_file("config.yaml")
    settings = Settings('my_settings')
    processor = ToyBinaryClassificationDataProcessor(config.data.processor)
    provider = MyDataProvider(config.data.provider, processor.data_path, processor.splits_path)
    exp = Experiment(config, Path("exp"), provider.dim_features, provider.dim_target)

    for fold_data in provider:
        if len(fold_data) == 4:
            outfold, infold, trloader, valoader = fold_data
            exp.run_training(trloader, valoader)
        if len(fold_data) == 2:
            outfold, teloader = fold_data
            exp.run_evaluation(teloader, exp.ckpts_dir)
