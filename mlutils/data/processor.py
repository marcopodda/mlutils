from pathlib import Path
import numpy as np

import torch

from mlutils.settings import Settings
from mlutils.util.module_loading import import_string
from mlutils.util.os import get_or_create_dir, dir_is_empty


class DataProcessor:
    def __init__(self, config, splitter_class):
        self.config = config
        self.settings = Settings()
        self.root = Path(self.settings.DATA_DIR) / config.dataset_name
        self.splitter_class = splitter_class

        self.raw_dir = get_or_create_dir(self.root / self.settings.RAW_DIR)
        self.processed_dir = get_or_create_dir(self.root / self.settings.PROCESSED_DIR)
        self.splits_dir = get_or_create_dir(self.root / self.settings.SPLITS_DIR)

        self._fetch_data()
        data, targets = self._process_data()
        self._split_data(data, targets=targets)

    def _fetch_data(self):
        raise NotImplementedError

    def _process_data(self):
        raise NotImplementedError

    def _split_data(self, data=None, targets=None):
        raise NotImplementedError

    @property
    def data_path(self):
        if dir_is_empty(self.processed_dir):
            raise ValueError("Dataset is not preprocessed!")
        return self.processed_dir / self.settings.DATASET_FILENAME

    @property
    def splits_path(self):
        if dir_is_empty(self.splits_dir):
            raise ValueError("Splits are not calculated!")
        return self.splits_dir / self.settings.SPLITS_FILENAME


class ToyDataProcessor(DataProcessor):
    def _fetch_data(self):
        pass

    def _process_data(self):
        data, targets = None, None
        if dir_is_empty(self.processed_dir):
            data_generator = import_string(f'sklearn.datasets.make_{self.task}')
            features, targets = data_generator(**self.config.params)
            data = np.hstack([features, targets.reshape(-1, 1)])
            torch.save(data, self.processed_dir / self.settings.DATASET_FILENAME)
        return data, targets

    def _split_data(self, data, targets=None):
        if dir_is_empty(self.splits_dir) and data is not None:
            indices = range(len(data))
            splitter = self.splitter_class(**self.config.splitter.params)
            splitter.split(indices, stratification=targets)
            splitter.save(self.splits_dir / self.settings.SPLITS_FILENAME)


class ToyBinaryClassificationDataProcessor(ToyDataProcessor):
    task = 'classification'


class ToyRegressionDataProcessor(ToyDataProcessor):
    task = 'regression'
