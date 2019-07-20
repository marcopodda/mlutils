import torch


def get_device(config):
    use_cuda = config.device in ["cuda", "gpu"] and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def entry2log(name):
    if name in ['loss', 'test', 'time']:
        return name
    if name == 'training':
        return 'train'
    if name == 'validation':
        return 'val'
    return name[:3]


def pretty_print(result):
    log = []
    training_keys = [k for (k, v) in result.items() if 'training' in k]
    validation_keys = [k for (k, v) in result.items() if 'validation' in k]
    test_keys = [k for (k, v) in result.items() if 'test' in k]

    for train_key, val_key in zip(training_keys, validation_keys):
        name = "_".join([entry2log(k) for k in train_key.split("_")])
        value = f"{result[train_key]:.6f}"
        log.append(f"{name}: {value}")
        name = "_".join([entry2log(k) for k in val_key.split("_")])
        value = f"{result[val_key]:.6f}"
        log.append(f"{name}: {value}")

    for test_key in test_keys:
        name = "_".join([entry2log(k) for k in test_key.split("_")])
        value = f"{result[test_key]:.6f}"
        log.append(f"{name}: {value}")

    return "  ".join(log)
