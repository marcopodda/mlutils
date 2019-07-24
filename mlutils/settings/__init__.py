from mlutils.util.serialize import save_yaml
from simple_settings import LazySettings


def build_config(settings_file):
    settings = LazySettings('mlutils.settings.defaults', settings_file)
    save_yaml(settings.CONFIG, "config.yaml")
