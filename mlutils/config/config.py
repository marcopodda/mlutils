import itertools

from mlutils.config import consts as const
from mlutils.util import check
from mlutils.util.serialize import load_yaml, save_yaml


NOT_A_LIST_ERROR_MSG = """
The value associated with parameter '{key}' must be a list.
"""


class ConfigError(Exception):
    pass


class LoadMixin:
    @classmethod
    def from_file(cls, path):
        config_dict = load_yaml(path)
        return cls(**config_dict)


class Config(LoadMixin):
    def __init__(self, **options):
        for name, value in const.DEFAULTS.items():
            setattr(self, name, value)

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
        if check.is_dictlike(value):
            return Config(**value)
        if check.is_nonempty_sequence_of_dicts(value):
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
            if not check.is_iterable(value):
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
