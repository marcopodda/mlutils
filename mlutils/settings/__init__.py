import warnings
from simple_settings import LazySettings


class Settings(object):
    _instance = None

    def __new__(cls, settings_file=None):
        if Settings._instance is None:
            if settings_file is None:
                warnings.warn("Initializing experiment with defaults values.")
                settings = LazySettings('mlutils.settings.defaults')
            else:
                settings = LazySettings('mlutils.settings.defaults', settings_file)
            Settings._instance = settings
        return Settings._instance
