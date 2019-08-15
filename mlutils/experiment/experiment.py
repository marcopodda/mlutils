from pathlib import Path

from mlutils.config import Config
from mlutils.settings import Settings
from mlutils.core.logging import Logger
from mlutils.util.os import get_or_create_dir
from mlutils.util.module_loading import import_class


class Experiment:
    def __init__(self,
                 config,
                 exp_path,
                 dim_features,
                 dim_target,
                 engine_class=None,
                 model_class=None,
                 criterion_class=None):

        self.config = config
        self.settings = Settings()
        self.root = exp_path
        self.dim_features = dim_features
        self.dim_target = dim_target

        self.engine_class = engine_class or import_class(
            self.config.engine, default=None)
        self.model_class = model_class or import_class(
            self.config.engine.model, default=None)
        self.criterion_class = criterion_class or import_class(
            self.config.engine.criterion, default=None)

        self.ckpts_dir = get_or_create_dir(self.root / self.settings.CKPTS_DIR)
        self.logdir = get_or_create_dir(self.root / self.settings.LOGS_DIR)

    def get_engine(self, ckpt_dir=None):
        engine = self.engine_class(
            self.config.engine,
            self.model_class,
            self.criterion_class,
            self.dim_features,
            self.dim_target,
            self.ckpts_dir,
            self.logdir)

        engine.set_callbacks(self.config.engine)

        if ckpt_dir is not None:
            engine.load(ckpt_dir, best=True)

        return engine

    def run_training(self, train_loader, val_loader=None):
        engine = self.get_engine()
        train_results = engine.fit(train_loader,
                                        val_loader=val_loader,
                                        num_epochs=self.config.engine.num_epochs)
        return train_results

    def run_evaluation(self, test_loader, ckpt_dir):
        engine = self.get_engine(ckpt_dir)
        eval_results = engine.evaluate(test_loader)
        return eval_results
