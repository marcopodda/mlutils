import numpy as np
from collections import defaultdict

from mlutils.settings import Settings
from mlutils.util.module_loading import import_class
from mlutils.util.os import get_or_create_dir
from mlutils.experiment import Experiment


class ModelSelector:
    def __init__(self,
                 configs,
                 path,
                 provider_class,
                 processor_class,
                 select_on="validation_loss",
                 mode="min"):
        self.configs = configs
        self.num_configs = len(configs)
        self.settings = Settings()
        self.root = path
        self.select_on = select_on
        self.mode = mode
        self.processor_class = processor_class
        self.provider_class = provider_class

    def run(self, outer_fold=0):
        config_results = []
        for i, config in enumerate(self.configs):
            processor = self.processor_class(config.processor)
            provider = self.provider_class(config.provider, processor.data_path, processor.splits_path)
            path = get_or_create_dir(self.root / f"CONFIG_{i}")
            config.save(path / self.settings.CONFIG_FILENAME)
            exp = Experiment(config, path, provider.dim_features, provider.dim_target)

            model_results = []
            for train_loader, val_loader in provider.get_model_selection_fold(outer_fold):
                result = exp.run_training(train_loader, val_loader)
                model_results.append(result)
            config_results.append(model_results)

        averages = self.average_results(config_results, provider.num_inner_folds)
        best_idx = self.select_best(averages)
        return self.configs[best_idx]

    def average_results(self, config_results, num_inner_folds):
        averages = defaultdict(list)
        for config_result in config_results:
            for k in config_result[0].keys():
                scores = []
                for fold in range(num_inner_folds):
                    scores.append(config_result[fold][k]['best'])
                averages[f"{k}_mean"].append(np.mean(scores))
                averages[f"{k}_std"].append(np.std(scores))
        return averages

    def select_best(self, averages):
        fun = np.argmin if self.mode == 'min' else np.argmax
        return fun(averages[f"{self.select_on}_mean"])







