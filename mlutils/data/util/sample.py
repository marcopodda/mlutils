from mlutils.settings import defaults


class Sample:
    def __init__(self, pattern_dict):
        if isinstance(pattern_dict, Sample):
            self.__dict__ = pattern_dict.__dict__
        else:
            self.x = pattern_dict[defaults.FEATURES_NAME]

            keys = [k for k in pattern_dict if k != defaults.FEATURES_NAME]
            for k in keys:
                setattr(self, k, pattern_dict[k])

        self._validate()

    def _validate(self):
        assert defaults.FEATURES_NAME in self

    def __getitem__(self, name):
        return getattr(self, name)

    def __iter__(self):
        return iter(self.__dict__)

    def __contains__(self, key):
        return key in self.__dict__

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()
