import os
import torch
import numpy as np
from pathlib import Path

from .datasets import Dataset

from utils.os import maybe_makedir
from utils.serialize import load_yaml, save_yaml


class DataManager:
    def __init__(self, root, dataset_name):
        self.root = Path(root) / dataset_name
        self.dataset_name = dataset_name

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

        self.inputs, self.targets = self._load_data()

    def _fetch_data(self, dest_dir):
        raise NotImplementedError

    def _process_data(self, dest_dir):
        raise NotImplementedError

    def _load_data(self):
        raise NotImplementedError

    def split_data(self, splitter):
        split_names = ['training', 'validation', 'test']
        splits_dir_path = self.root / "splits"
        if not (splits_dir_path).exists():
            os.makedirs(splits_dir_path)
            indices = range(len(self.inputs))
            splitter.split(indices, stratification=self.targets)

            for split in split_names:
                save_yaml(splitter.get_split(split), splits_dir_path / f"{split}.yaml")

        for split in split_names:
            setattr(self, f"{split}_split", load_yaml(splits_dir_path / f"{split}.yaml"))

    def get_dataset(self, name, outer_fold=0, inner_fold=0):
        indices = getattr(self, f"{name}_split")[outer_fold][inner_fold]
        return Dataset(self.inputs, self.targets, indices)



