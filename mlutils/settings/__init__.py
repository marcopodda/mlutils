import warnings
from mlutils.util.serialize import save_yaml
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
            save_yaml(settings.CONFIG, "config.yaml")
            Settings._instance = settings
        return Settings._instance