import itertools
from utils.serialize import load_yaml, save_yaml


NOT_A_LIST_ERROR_MSG = """
The value associated with parameter '{key}' must be a list.
"""


DEFAULT_CONFIG = {
    'model_class': 'modules.models.MLP',
    'criterion_class': 'modules.criterions.BCE',
    'optimizer_class': 'torch.optim.Adam',
    'optimizer_params': {
        'lr': 0.001
    },
    'dim_hidden': 128,
    'data_root': "DATA",
    'dataset_name': 'random',
    'splitter_class': 'data.splitters.HoldoutSplitter',
    'splitter_params': {},
    'dataloader_params': {
        'batch_size': 32,
        'shuffle': True
    },
    'max_epochs': 10,
    'scheduler_class': 'torch.optim.lr_scheduler.StepLR',
    'scheduler_params': {
        'gamma': 0.5,
        'step_size': 30
    },
    'grad_clip_func': 'torch.nn.utils.clip_grad.clip_grad_norm_',
    'grad_clip_params': {
        'max_norm': 1.0,
        'norm_type': 2
    },
    'device': 'cpu',
    'model': {
        'class_name': 'modules.models.MLP',
        'params': {
            'dim_hidden': 128
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
        'dataset': {
            'class_name': 'data.datasets.RandomDataset',
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
        'scheduler': {
            'class_name': 'torch.optim.lr_scheduler.StepLR',
            'params': {
                'gamma': 0.5,
                'step_size': 30
            },
        }
    },
    'event_handlers': [
        {
            'class_name': 'core.loggers.LossLogger',
            'params': {}
        },
        {
            'class_name': 'core.gradient_clipping.GradientClipper',
            'params': {
                'func': 'torch.nn.utils.clip_grad.clip_grad_norm_',
                'args': {
                    'max_norm': 1.0
                }
            }
        }
    ],
    'metrics': [
        {
            'class_name': 'core.loggers.TrainingMetricLogger',
            'params': {
                'metrics': ['core.metrics.BinaryAccuracy']
            }
        },
        {
            'class_name': 'core.loggers.ValidationMetricLogger',
            'params': {
                'metrics': ['core.metrics.BinaryAccuracy']
            }
        },
    ]
}


class ConfigError(Exception):
    pass


class LoadMixin:
    @classmethod
    def from_file(cls, path):
        config_dict = load_yaml(path)
        return cls(**config_dict)


class Config(LoadMixin):
    def __init__(self, **config_dict):
        for name, value in DEFAULT_CONFIG.items():
            setattr(self, name, value)

        for name, value in config_dict.items():
            setattr(self, name, value)

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return name in self.__dict__

    def get(self, *keys):
        config_dict = self.__dict__.copy()
        for key in keys:
            config_dict = config_dict[key]

        if isinstance(config_dict, list):
            return [Config(**cd) for cd in config_dict]

        if not isinstance(config_dict, dict):
            raise ValueError(f"'{config_dict}' is not a dictionary.")

        return Config(**config_dict)

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

