import torch

from mlutils.settings import defaults

def get_device(config):
    use_cuda = config.device in ["cuda", "gpu"] and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def is_training_fold(loader_data):
    return len(loader_data) == 4


def is_evaluation_fold(loader_data):
    return len(loader_data) == 2


def _entry2log(name):
    if name in ['loss', defaults.TEST, 'time']:
        return name
    if name == defaults.TRAINING:
        return 'train'
    if name == defaults.VALIDATION:
        return 'val'
    return name[:3]


def pretty_print(result):
    log_msg = []
    training_keys = [k for (k, v) in result.items() if defaults.TRAINING in k]
    validation_keys = [k for (k, v) in result.items() if defaults.VALIDATION in k]
    test_keys = [k for (k, v) in result.items() if defaults.TEST in k]

    for train_key, val_key in zip(training_keys, validation_keys):
        name = "_".join([_entry2log(k) for k in train_key.split("_")])
        value = f"{result[train_key]:.6f}"
        log_msg.append(f"{name}: {value}")
        name = "_".join([_entry2log(k) for k in val_key.split("_")])
        value = f"{result[val_key]:.6f}"
        log_msg.append(f"{name}: {value}")

    for test_key in test_keys:
        name = "_".join([_entry2log(k) for k in test_key.split("_")])
        value = f"{result[test_key]:.6f}"
        log_msg.append(f"{name}: {value}")

    return "\t".join(log_msg)
