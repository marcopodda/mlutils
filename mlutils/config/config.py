import itertools

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
        self._dict = dict(**options)

    def update(self, **options):
        for name, value in options.items():
            setattr(self, name, value)

    def __getitem__(self, name):
        return getattr(self, name)

    def __setitem__(self, name, value):
        self._dict[name] = value

    def __getattr__(self, name):
        value = self._dict[name]

        if check.is_dictlike(value):
            return Config(**value)

        if check.is_nonempty_sequence_of_dicts(value):
            return [Config(**v) for v in value if v != {}]

        return value

    def __contains__(self, name):
        return name in self._dict and self._dict[name] not in [None, {}, []]

    def save(self, path):
        save_yaml(self._dict, path)

    def keys(self):
        return self._dict.keys()

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
