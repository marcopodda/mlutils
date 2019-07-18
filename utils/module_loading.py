import copy
import os
from importlib import import_module
from importlib.util import find_spec as importlib_find


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


def dynamic_class_load(obj_config, main_param=None):
    obj_class = import_string(obj_config.class_name)
    if main_param is not None:
        return obj_class(main_param, **obj_config.params)
    return obj_class(**obj_config.params)