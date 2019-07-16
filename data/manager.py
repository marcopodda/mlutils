import os
import operator
import numpy as np
from pathlib import Path

import torch
from torch.utils import data

from utils.module_loading import import_string
from utils.serialize import load_yaml, save_yaml
from .splitters import HoldoutSplitter


class DataManager:
    def __init__(self, config):
        self.config = config
        self.root = Path(config.data_root) / config.dataset_name

        raw_dir_path = self.root / "raw"
        if not (raw_dir_path).exists():
            os.makedirs(raw_dir_path)
            self._fetch_data(raw_dir_path)
        self.raw_dir = raw_dir_path

        processed_dir_path = self.root / "processed"
        if not (processed_dir_path).exists():
            os.makedirs(processed_dir_path)
            self._process_data(processed_dir_path)
        self.processed_dir = processed_dir_path

        self.dataset = self._load_data()

        splits_dir_path = self.root / "splits"
        if not (splits_dir_path).exists():
            os.makedirs(splits_dir_path)
            self._split_data(splits_dir_path)
        self.splits_dir = splits_dir_path

        self.splits = self._load_splits()
        self.outer_folds = len(self.splits['test'])
        self.inner_folds = len(self.splits['training'][0])

    def _fetch_data(self, raw_dir):
        raise NotImplementedError

    def _process_data(self, processed_dir):
        raise NotImplementedError

    def _load_data(self):
        return torch.load(self.processed_dir / "dataset.pt")

    def _split_data(self, splits_dir_path):
        splitter_class = import_string(self.config.splitter_class)
        splitter = splitter_class(**self.config.splitter_params)
        indices, targets = self.dataset.indices, self.dataset.targets
        splits = splitter.split(indices, stratification=targets)
        save_yaml(splits, splits_dir_path / "splits.yaml")

    def _load_splits(self):
        return load_yaml(self.splits_dir / "splits.yaml")

    def get_loader(self, name, outer_fold=0, inner_fold=0):
        indices = self.splits[name][outer_fold][inner_fold]
        partition = data.Subset(self.dataset, indices)
        return data.DataLoader(partition, **self.config.dataloader_params)

    def get_data(self, name, outer_fold=0, inner_fold=0):
        indices = self.splits[name][outer_fold][inner_fold]
        partition = operator.itemgetter(*indices)(self.dataset._data)
        return partition

    @property
    def dim_input(self):
        return self.dataset.dim_input

    @property
    def dim_target(self):
        return self.dataset.dim_target





