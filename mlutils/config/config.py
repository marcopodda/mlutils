import itertools
from mlutils.util.serialize import load_yaml, save_yaml


NOT_A_LIST_ERROR_MSG = """
The value associated with parameter '{key}' must be a list.
"""


# DEFAULTS = {
#     'max_epochs': 10,
#     'device': 'cpu',
#     'model': {
#         'class_name': 'modules.models.MLP',
#         'params': {
#             'dim_layers': [128, 64]
#         }
#     },
#     'criterion': {
#         'class_name': 'modules.criterions.BCE',
#         'params': {}
#     },
#     'data': {
#        'processor':{
#             "root": "DATA",
#             "name": "dataset",
#             "raw_dir_name": "raw",
#             "processed_dir_name": "processed",
#             "dataset_filename": "dataset.pt",
#             'splitter': {
#                 "splits_dir_name": "splits",
#                 "splits_filename": "splits.yaml",
#                 'class_name': 'data.splitters.HoldoutSplitter',
#                 'params': {}
#             },
#             "params": {}
#         },
#         'provider': {
#             'dataset': {
#                 'class_name': 'data.datasets.ToyClassificationDataset',
#                 'params': {'num_features': 32}
#             },
#             'loader': {
#                 'class_name': 'torch.utils.data.DataLoader',
#                 'params': {
#                     'batch_size': 32,
#                     'shuffle': True
#                 }
#             }
#         }
#     },
#     'optimizer': {
#         'class_name': 'torch.optim.Adam',
#         'params': {'lr': 0.001},
#     },
#     'callbacks': {
#         'metrics': [
#             {'class_name': 'core.metrics.BinaryAccuracy'}
#         ]
#     }
# }

TEST_CONFIG = {
    'max_epochs': 10,
    'device': 'cpu',
    'model': {
        'name': 'mlp',
        'class_name': 'mlutils.modules.models.MLP',
        'params': {'dim_layers': [128, 64]}
    },
    'criterion': {
        'name': 'cross_entropy',
        'class_name': 'mlutils.modules.criterions.CrossEntropy'
    },
    'data': {
        'name': 'toy_classification',
        'dataset': {
            'class_name': 'mlutils.data.datasets.ToyClassificationDataset',
            'params': {'n_samples': 10000, 'n_features': 16, 'n_informative': 8, 'n_classes': 3}
        },
        'splitter': {
            'class_name': 'mlutils.data.splitters.HoldoutSplitter',
            'params': {'stratified': True}
        },
        'loader': {
            'class_name': 'torch.utils.data.DataLoader',
            'params': {'batch_size': 32, 'shuffle': True}
        }
    },
    'optimizer': {
        'name': 'adam',
        'class_name': 'torch.optim.Adam',
        'params': {'lr': 0.001},
        'scheduler': {
            'class_name': 'torch.optim.lr_scheduler.StepLR',
            'params': {'gamma': 0.5, 'step_size': 30},
        },
        # 'gradient_clipper': {
        #     'class_name': 'core.optimizer.GradientClipper',
        #     'params': {
        #         'func': 'torch.nn.utils.clip_grad.clip_grad_norm_',
        #         'args': {'max_norm': 1.0}
        #     }
        # },
    },
    'callbacks': {
        'metrics': [
            {'class_name': 'mlutils.core.metrics.MulticlassAccuracy'},
            # {'class_name': 'mlutils.core.metrics.AUC'},
            # {'class_name': 'core.metrics.Time'}
        ],
        'early_stopper': {
            'class_name': 'mlutils.core.early_stopping.PatienceEarlyStopper',
            'params': {'patience': 10}
        },
        'model_saver': {'class_name': 'mlutils.core.saver.ModelSaver'},
        'loggers': [{'class_name': 'mlutils.core.loggers.CSVLogger'}]
    }
}


class ConfigError(Exception):
    pass


class LoadMixin:
    @classmethod
    def from_file(cls, path):
        config_dict = load_yaml(path)
        return cls(**config_dict)


class Config(LoadMixin):
    def __init__(self, **options):
        for name, value in options.items():
            setattr(self, name, value)

    def update(self, **options):
        for name, value in options.items():
            setattr(self, name, value)

    def __getitem__(self, name):
        return self.__getattribute__(name)

    def __setitem__(self, name, value):
        setattr(self, name, value)

    def __getattr__(self, name):
        return self.__getattribute__(name)

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if isinstance(value, dict):
            return Config(**value)
        if isinstance(value, list) and value != [] and isinstance(value[0], dict):
            return [Config(**v) for v in value]
        return value

    def __contains__(self, name):
        obj_dict = object.__getattribute__(self, "__dict__")
        return name in obj_dict and obj_dict[name] not in [None, {}, []]

    def save(self, path):
        obj_dict = object.__getattribute__(self, "__dict__")
        save_yaml(obj_dict, path)


class ModelSelectionConfig(LoadMixin):
    def __init__(self, **ms_dict):
        self._validate(ms_dict)
        self._dict = ms_dict

    def _validate(self, ms_dict):
        for name, value in ms_dict.items():
            if not isinstance(value, list):
                msg = NOT_A_LIST_ERROR_MSG.format(key=name)
                raise ConfigError(msg)

    def __iter__(self):
        items = sorted(self._dict.items())
        if not items:
            yield {}
        else:
            keys, values = zip(*items)
            for v in itertools.product(*values):
                params = dict(zip(keys, v))
                yield params
