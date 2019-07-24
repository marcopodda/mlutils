import yaml


def load_yaml(path):
    return yaml.load(open(path, "r"), Loader=yaml.FullLoader)


def save_yaml(obj, path):
    return yaml.dump(obj, open(path, "w"))
