import itertools
from utils.serialize import load_yaml, save_yaml


NOT_A_LIST_ERROR_MSG = """
The value associated with parameter '{key}' must be a list.
"""

class ConfigError(Exception):
    pass


class LoadMixin:
    @classmethod
    def from_file(cls, path):
        config_dict = load_yaml(path)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


class Config(LoadMixin):
    def __init__(self, **params):
        self.update(**params)

    def __getitem__(self, name):
        return getattr(self, name)

    def update(self, **params):
        for name, value in params.items():
            setattr(self, name, value)

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

