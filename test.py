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
    selector = ModelSelector(configs, Path("MODEL_SELECTION"), processor_class=ToyBinaryClassificationDataProcessor, provider_class=MyDataProvider)
    print(selector.run())

    # for fold_data in provider:
    #     ckpts_dir = None

    #     if len(fold_data) == 4:
    #         outfold, infold, trloader, valoader = fold_data
    #         exp = Experiment(config, Path(settings.EXP_DIR) / f"exp_{outfold}_{infold}", provider.dim_features, provider.dim_target)
    #         exp.run_training(trloader, valoader)
    #         ckpts_dir = exp.ckpts_dir

    #     if len(fold_data) == 2:
    #         outfold, teloader = fold_data
    #         exp = Experiment(config, Path(settings.EXP_DIR) / f"exp_{outfold}", provider.dim_features, provider.dim_target)
    #         exp.run_evaluation(teloader, ckpts_dir)