from pathlib import Path
import numpy as np

import torch

from mlutils.config import Config
from mlutils.util.module_loading import import_string
from mlutils.util.os import get_or_create_dir, dir_is_empty

from .splitter import HoldoutSplitter


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.root = Path(config.root) / config.name

        self.raw_dir = get_or_create_dir(self.root / config.raw_dir_name)
        self.processed_dir = get_or_create_dir(self.root / config.processed_dir_name)
        self.splits_dir = get_or_create_dir(self.root / config.splitter.splits_dir_name)

        self._fetch_data()
        data, targets = self._process_data()
        self._split_data(data=data, targets=targets)

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
        return self.processed_dir / self.config.dataset_filename

    @property
    def splits_path(self):
        if dir_is_empty(self.splits_dir):
            raise ValueError("Splits are not calculated!")
        return self.splits_dir / self.config.splitter.splits_filename


class ToyDataProcessor(DataProcessor):
    def _fetch_data(self):
        pass

    def _process_data(self):
        data, targets = None, None
        if dir_is_empty(self.processed_dir):
            fn = import_string(f'sklearn.datasets.make_{self.task}')
            features, targets = fn(**self.config.params)
            data = np.hstack(features, targets.reshape(-1, 1))
            torch.save(data, self.data_path)
        return data, targets

    def _split_data(self, data=None, targets=None):
        if dir_is_empty(self.splits_dir) and data is not None:
            splitter = HoldoutSplitter(**self.config.splitter.params)
            splitter.split(range(len(data)), stratification=targets)
            splitter.save(self.splits_path)


class ToyClassificationDataProcessor(ToyDataProcessor):
    task = 'classification'


class ToyRegressionDataProcessor(ToyDataProcessor):
    task = 'regression'