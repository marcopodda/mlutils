from torch.utils.data import Subset
from mlutils.settings import Settings
from mlutils.util.serialize import load_yaml
from mlutils.util.module_loading import import_class


class DataProvider:
    def __init__(self,
                 config,
                 data_path,
                 splits_path,
                 dataset_class=None,
                 loader_class=None):

        self.config = config
        self.settings = Settings()

        self.loader_class = loader_class or import_class(
            config.loader, default=self.settings.LOADER)

        dataset_class = dataset_class or import_class(
            config.dataset, default=None)
        self.dataset = dataset_class(data_path)

        self.splits = load_yaml(splits_path)

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
        return len(self.splits[self.settings.TEST])

    @property
    def num_inner_folds(self):
        return len(self.splits[self.settings.TRAINING][0])

    def __iter__(self):
        for outer_fold in range(self.num_outer_folds):
            for inner_fold in range(self.num_inner_folds):
                training_fold_loader = self.get_loader(self.settings.TRAINING, outer_fold, inner_fold)
                validation_fold_loader = self.get_loader(self.settings.VALIDATION, outer_fold, inner_fold)
                yield outer_fold, inner_fold, training_fold_loader, validation_fold_loader

            test_fold_loader = self.get_loader(self.settings.TEST, outer_fold, inner_fold)
            yield outer_fold, test_fold_loader
