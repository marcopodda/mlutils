import torch

from pathlib import Path

from mlutils.config import Config
from mlutils.settings import defaults
from mlutils.data.processor import DataProcessor
from mlutils.data.dataset import ToyBinaryClassificationDataset
from mlutils.data.provider import DataProvider
from mlutils.data.splitter import HoldoutSplitter
from mlutils.core.engine import Engine
from mlutils.core.logging import Logger
from mlutils.util.os import get_or_create_dir
from mlutils.util.module_loading import load_class
from mlutils.util.training import is_training_fold, is_evaluation_fold
from mlutils.modules.models import MLP
from mlutils.modules.criterions import BinaryCrossEntropy


class Experiment:
    @property
    def name(self):
        if not hasattr(self, '_name'):
            self._name = defaults.EXP_NAME
        return self._name

    def __init__(self,
                 config_file,
                 processor_class=DataProcessor,
                 splitter_class=HoldoutSplitter,
                 provider_class=DataProvider,
                 dataset_class=ToyBinaryClassificationDataset,
                 loader_class=torch.utils.data.DataLoader,
                 engine_class=Engine,
                 model_class=MLP,
                 criterion_class=BinaryCrossEntropy):

        self.config = Config.from_file(config_file)
        self.root = self.get_exp_root_folder()



        self.processor_class = processor_class
        self.splitter_class = splitter_class
        self.provider_class = provider_class
        self.dataset_class = dataset_class
        self.loader_class = loader_class
        self.engine_class = engine_class
        self.model_class = model_class
        self.criterion_class = criterion_class

        self.define_folder_structure()

        self.logger = Logger(self.logs_dir)

    def define_folder_structure(self):
        self.ckpts_dir = get_or_create_dir(self.root / defaults.CKPTS_DIR)
        self.config_dir = get_or_create_dir(self.root / defaults.CONFIG_DIR)
        self.data_dir = get_or_create_dir(self.root / defaults.EXPDATA_DIR)
        self.results_dir = get_or_create_dir(self.root / defaults.RESULTS_DIR)
        self.logs_dir = get_or_create_dir(self.root / defaults.LOGS_DIR)

    def define_callbacks(self):
        return []

    def get_exp_root_folder(self):
        path = Path(defaults.EXP_DIR) / self.name
        return get_or_create_dir(path)

    def run(self, resume=False):
        processor = self.processor_class(
            self.config.data.processor,
            self.root,
            self.splitter_class)

        provider = self.provider_class(
            self.config.data.provider,
            self.dataset_class,
            self.loader_class,
            processor.data_path,
            processor.splits_path)

        engine = self.engine_class(
            self.config.engine,
            self.model_class,
            self.criterion_class,
            provider.dim_features,
            provider.dim_target)

        engine.set_callbacks(self.config.engine)

        if resume is True:
            engine.load(self.ckpts_dir, best=False)

        for loader_data in provider:
            if is_training_fold(loader_data):
                outer_fold, inner_fold, training_loader, validation_loader = loader_data
                engine.fit(training_loader, val_loader=validation_loader, num_epochs=self.config.engine.num_epochs)

            if is_evaluation_fold(loader_data):
                outer_fold, test_loader = loader_data
                engine.evaluate(test_loader, path=self.ckpts_dir)


# def ModelSelection(Experiment):
#     exp_type = "MODEL_SELECTION"
#     splitter_class = CVHoldoutSplitter


# def ModelEvaluation(Experiment):
#     exp_type = "MODEL_EVALUATION"
#     splitter_class = HoldoutSplitter
