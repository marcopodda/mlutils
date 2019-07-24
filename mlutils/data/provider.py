from torch.utils.data import Subset
from mlutils.settings import defaults
from mlutils.util.module_loading import load_class
from mlutils.util.serialize import load_yaml
from .dataset import FileDataset


class DataProvider:
    def __init__(self, config, dataset_class, loader_class, data_path, splits_path):
        self.config = config
        self.dataset = dataset_class(data_path)
        self.splits = load_yaml(splits_path)
        self.loader_class = loader_class

    def get_loader(self, split_name, outer_fold=0, inner_fold=0):
        indices = self.splits[split_name][outer_fold][inner_fold]
        partition = Subset(self.dataset, indices)
        loader = self.loader_class(partition, **self.config.loader.params)
        return loader

    @property
    def dim_features(self):
        raise NotImplementedError

    @property
    def dim_target(self):
        raise NotImplementedError

    @property
    def num_outer_folds(self):
        return len(self.splits[defaults.TEST])

    @property
    def num_inner_folds(self):
        return len(self.splits[defaults.TRAINING][0])

    def __iter__(self):
        for outer_fold in range(self.num_outer_folds):
            for inner_fold in range(self.num_inner_folds):
                training_fold_loader = self.get_loader(defaults.TRAINING, outer_fold, inner_fold)
                validation_fold_loader = self.get_loader(defaults.VALIDATION, outer_fold, inner_fold)
                yield outer_fold, inner_fold, training_fold_loader, validation_fold_loader

            test_fold_loader = self.get_loader(defaults.TEST, outer_fold, inner_fold)
            yield outer_fold, test_fold_loader
