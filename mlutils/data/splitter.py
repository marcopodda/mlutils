import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit

from mlutils.settings import Settings
from mlutils.util.serialize import save_yaml


class Splitter:
    def __init__(self, stratified):
        assert hasattr(self, 'outer_splitter'), "You must define attribute 'outer_splitter'."
        assert hasattr(self, 'inner_splitter'), "You must define attribute 'inner_splitter'."
        self.settings = Settings()
        self.initialized = False
        self.stratified = stratified

    def _validate_splits(self, splits):
        assert all(k in splits for k in self.settings.LEARNING_MODES), \
            f'Split names must be equal to {self.settings.LEARNING_MODES}.'
        assert all(isinstance(splits[k], list) for k in self.settings.LEARNING_MODES), \
            f'Splits must be lists.'
        assert all(len(splits[k]) > 0 for k in self.settings.LEARNING_MODES), \
            'Splits length must be > 0.'

    def split(self, indices, stratification=None):
        if self.initialized:
            raise Exception('Multiple splitting is not allowed.')

        if self.stratified is True and stratification is None:
            raise ValueError("You must provide a stratification array if 'stratified' is True.")

        indices = np.array(indices)
        splits = {self.settings.TRAINING: [], self.settings.VALIDATION: [], self.settings.TEST: []}

        if stratification is not None:
            stratification = np.array(stratification)

        outer_splitter = self.outer_splitter.split(indices, y=stratification)
        for outer_train_idx, outer_test_idx in outer_splitter:
            splits[self.settings.TEST].append([indices[outer_test_idx].tolist()])

            if stratification is not None:
                inner_stratification = stratification[outer_train_idx]
            else:
                inner_stratification = None

            train_inner_folds, val_inner_folds = [], []
            inner_splitter = self.inner_splitter.split(indices[outer_train_idx], y=inner_stratification)
            for inner_train_idx, inner_val_idx in inner_splitter:
                train_inner_folds.append(outer_train_idx[inner_train_idx].tolist())
                val_inner_folds.append(outer_train_idx[inner_val_idx].tolist())

            splits[self.settings.TRAINING].append(train_inner_folds)
            splits[self.settings.VALIDATION].append(val_inner_folds)

        self.splits = splits
        self.initialized = True

    def get_split(self, partition, outer_fold=0, inner_fold=0):
        assert self.initialized, 'Not initialized.'
        assert partition in self.settings.LEARNING_MODES, f'Unknown partition {partition}.'
        assert outer_fold < self.outer_k, f"'outer_fold' must be less than {self.outer_k}, but got {outer_fold}."
        assert inner_fold < self.inner_k, f"'inner_fold' must be less than {self.inner_k}, but got {inner_fold}."
        return self.splits[partition][outer_fold][inner_fold]

    def save(self, path):
        save_yaml(self.splits, path)


class HoldoutSplitter(Splitter):
    def __init__(self, stratified=False, test_size=0.2):
        if stratified:
            self.outer_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
            self.inner_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        else:
            self.outer_splitter = ShuffleSplit(n_splits=1, test_size=test_size)
            self.inner_splitter = ShuffleSplit(n_splits=1, test_size=test_size)

        super().__init__(stratified=stratified)


class CVHoldoutSplitter(Splitter):
    def __init__(self, stratified=False, test_size=0.2, inner_folds=5):
        if stratified:
            self.outer_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
            self.inner_splitter = StratifiedKFold(n_splits=inner_folds)
        else:
            self.outer_splitter = ShuffleSplit(n_splits=1, test_size=test_size)
            self.inner_splitter = KFold(n_splits=inner_folds)

        super().__init__(stratified=stratified)


class NestedHoldoutSplitter(Splitter):
    def __init__(self, stratified=False, test_size=0.2, outer_folds=5):
        if stratified:
            self.outer_splitter = StratifiedKFold(n_splits=outer_folds)
            self.inner_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        else:
            self.outer_splitter = KFold(n_splits=outer_folds)
            self.inner_splitter = ShuffleSplit(n_splits=1, test_size=test_size)

        super().__init__(stratified=stratified)


class NestedCVSplitter(Splitter):
    def __init__(self, stratified=False, outer_folds=5, inner_folds=3):
        if stratified:
            self.outer_splitter = StratifiedKFold(n_splits=outer_folds)
            self.inner_splitter = StratifiedKFold(n_splits=inner_folds)
        else:
            self.outer_splitter = KFold(n_splits=outer_folds)
            self.inner_splitter = KFold(n_splits=inner_folds)

        super().__init__(stratified=stratified)
