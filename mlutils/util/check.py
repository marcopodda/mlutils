from collections.abc import Iterable, Mapping


def is_iterable(obj):
    return isinstance(obj, Iterable)


def is_dictlike(obj):
    return isinstance(obj, Mapping)


def is_nonempty_sequence_of_dicts(obj):
    return is_iterable(obj) and len(obj) > 0 and all(is_dictlike(o) for o in obj)