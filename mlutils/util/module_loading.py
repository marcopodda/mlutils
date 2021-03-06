from importlib import import_module

from mlutils.config import Config


def import_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError as err:
        raise ImportError("%s doesn't look like a module path" % dotted_path) from err

    module = import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError as err:
        raise ImportError('Module "%s" does not define a "%s" attribute/class' % (
            module_path, class_name)
        ) from err


def load_class(obj, *args, **kwargs):
    if isinstance(obj, str):
        obj_class = import_string(obj)
        return obj_class(*args, **kwargs)

    if isinstance(obj, Config):
        obj_class = import_string(obj.class_name)
        if 'params' in obj:
            obj.params.update(**kwargs)
            kwargs = obj.params.asdict()

    return obj_class(*args, **kwargs)


def import_class(config, default):
    if 'class_name' in config:
        return import_string(config.class_name)

    if default is None:
        raise Exception("If class name is not provided in config, 'default' cannot be None")

    return import_string(default)
