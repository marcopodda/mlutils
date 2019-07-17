import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit, StratifiedShuffleSplit


class Splitter:
    def __init__(self, stratified=True):
        self.stratified = stratified

    def split(self, indices, stratification=None):
        indices = np.array(indices)

        splits ={
            'training': [],
            'validation': [],
            'test': []
        }

        if self.stratified is True and stratification is None:
            raise ValueError("You must provide a stratification array if 'stratified' is True.")

        if stratification is not None:
            stratification = np.array(stratification)

        outer_splitter = self.outer_splitter.split(indices, y=stratification)
        for outer_train_idx, outer_test_idx in outer_splitter:
            splits['test'].append([indices[outer_test_idx].tolist()])

            if stratification is not None:
                inner_stratification = stratification[outer_train_idx]
            else:
                inner_stratification = None

            train_inner_folds, val_inner_folds = [], []
            inner_splitter = self.inner_splitter.split(indices[outer_train_idx], y=inner_stratification)
            for inner_train_idx, inner_val_idx in inner_splitter:
                train_inner_folds.append(outer_train_idx[inner_train_idx].tolist())
                val_inner_folds.append(outer_train_idx[inner_val_idx].tolist())

            splits['training'].append(train_inner_folds)
            splits['validation'].append(val_inner_folds)

        return splits


class HoldoutSplitter(Splitter):
    def __init__(self, test_size=0.2, stratified=True):
        super().__init__(stratified=stratified)

        if stratified:
            self.outer_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
            self.inner_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        else:
            self.outer_splitter = ShuffleSplit(n_splits=1, test_size=test_size)
            self.inner_splitter = ShuffleSplit(n_splits=1, test_size=test_size)


class CVHoldoutSplitter(Splitter):
    def __init__(self, test_size=0.2, inner_folds=5, stratified=True):
        super().__init__(stratified=stratified)

        if stratified:
            self.outer_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
            self.inner_splitter = StratifiedKFold(n_splits=inner_folds)
        else:
            self.outer_splitter = ShuffleSplit(n_splits=1, test_size=test_size)
            self.inner_splitter = KFold(n_splits=inner_folds)


class NestedHoldoutSplitter(Splitter):
    def __init__(self, test_size=0.2, outer_folds=5, stratified=True):
        super().__init__(stratified=stratified)

        if stratified:
            self.outer_splitter = StratifiedKFold(n_splits=outer_folds)
            self.inner_splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        else:
            self.outer_splitter = KFold(n_splits=outer_folds)
            self.inner_splitter = ShuffleSplit(n_splits=1, test_size=test_size)


class NestedCVSplitter(Splitter):
    def __init__(self, outer_folds=5, inner_folds=3, stratified=True):
        super().__init__(stratified=stratified)

        if stratified:
            self.outer_splitter = StratifiedKFold(n_splits=outer_folds)
            self.inner_splitter = StratifiedKFold(n_splits=inner_folds)
        else:
            self.outer_splitter = KFold(n_splits=outer_folds)
            self.inner_splitter = KFold(n_splits=inner_folds)

