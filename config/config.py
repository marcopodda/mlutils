import itertools
from utils.serialize import load_yaml, save_yaml


NOT_A_LIST_ERROR_MSG = """
The value associated with parameter '{key}' must be a list.
"""


DEFAULTS = {
    'max_epochs': 10,
    'device': 'cpu',
    'model': {
        'class_name': 'modules.models.MLP',
        'params': {
            'dim_layers': [128, 64]
        }
    },
    'criterion': {
        'class_name': 'modules.criterions.BCE',
        'params': {}
    },
    'data': {
        'root': 'DATA',
        'name': 'random',
        'splitter': {
            'class_name': 'data.splitters.HoldoutSplitter',
            'params': {}
        },
        'loader': {
            'class_name': 'torch.utils.data.DataLoader',
            'params': {
                'batch_size': 32,
                'shuffle': True
            }
        }
    },
    'optimizer': {
        'class_name': 'torch.optim.Adam',
        'params': {'lr': 0.001},
    },
    'callbacks': {
        'metrics': [
            {'class_name': 'core.metrics.BinaryAccuracy'}
        ]
    }
}

TEST_CONFIG = {
    'max_epochs': 10,
    'device': 'cpu',
    'model': {
        'class_name': 'modules.models.MLP',
        'params': {'dim_layers': [128, 64]}
    },
    'criterion': {'class_name': 'modules.criterions.BCE'},
    'data': {
        'root': 'DATA',
        'name': 'random',
        'splitter': {'class_name': 'data.splitters.HoldoutSplitter'},
        'loader': {
            'class_name': 'torch.utils.data.DataLoader',
            'params': {'batch_size': 32, 'shuffle': True}
        }
    },
    'optimizer': {
        'class_name': 'torch.optim.Adam',
        'params': {'lr': 0.001},
        'scheduler': {
            'class_name': 'torch.optim.lr_scheduler.StepLR',
            'params': {'gamma': 0.5, 'step_size': 30},
        },
        'gradient_clipper': {
            'class_name': 'core.optimizer.GradientClipper',
            'params': {
                'func': 'torch.nn.utils.clip_grad.clip_grad_norm_',
                'args': {'max_norm': 1.0}
            }
        },
    },
    'callbacks': {
        'metrics': [
            {'class_name': 'core.metrics.BinaryAccuracy'},
            {'class_name': 'core.metrics.Time'}
        ],
        'early_stopper': {
            'class_name': 'core.early_stopping.PatienceEarlyStopper',
            # 'params': {'alpha': 0.2}
        },
        'model_saver': {'class_name': 'core.saver.ModelSaver'},
        'loggers': [{'class_name': 'core.loggers.CSVLogger'}]
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
    def __init__(self, use_defaults=True, **config_dict):
        if use_defaults:
            for name, value in TEST_CONFIG.items():
                setattr(self, name, value)

        for name, value in config_dict.items():
            setattr(self, name, value)

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return name in self.__dict__ and self.__dict__[name] not in [None, {}, []]

    def get(self, *keys):
        config_dict = self.__dict__.copy()
        for key in keys:
            config_dict = config_dict[key]

        if isinstance(config_dict, list):
            return [Config(use_defaults=False, **cd) for cd in config_dict]

        if not isinstance(config_dict, dict):
            raise ValueError(f"'{config_dict}' is not a dictionary.")

        return Config(use_defaults=False, **config_dict)

    def save(self, path):
        save_yaml(self.__dict__, path)


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

