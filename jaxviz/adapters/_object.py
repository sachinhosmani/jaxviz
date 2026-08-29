import contextlib
import threading
from collections import defaultdict

import jax

from .._module_scopes import module_scope_name


_SCOPE_LOCK = threading.RLock()


def _collect_value(value, name, module_type, id_to_name, modules):
    if isinstance(value, module_type):
        if id(value) in id_to_name:
            return
        id_to_name[id(value)] = name
        _collect(value, module_type, id_to_name, modules)
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_value(item, f"{name}_{index}", module_type, id_to_name, modules)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_value(item, f"{name}_{key}", module_type, id_to_name, modules)


def _collect(module, module_type, id_to_name, modules):
    modules.append(module)
    for attribute, value in vars(module).items():
        _collect_value(value, attribute, module_type, id_to_name, modules)


@contextlib.contextmanager
def object_module_scopes(model, module_type, root_name=None):
    id_to_name = {id(model): root_name or type(model).__name__}
    modules = []
    _collect(model, module_type, id_to_name, modules)
    module_types = {
        type(module) for module in modules
        if "__call__" in vars(type(module))
    }
    originals = {}
    invocation_counts = defaultdict(int)

    def make_wrapped(original):
        def wrapped(self, *args, **kwargs):
            name = id_to_name.get(id(self))
            if name is None:
                return original(self, *args, **kwargs)
            invocation_counts[id(self)] += 1
            scope_name = module_scope_name(name, invocation_counts[id(self)])
            with jax.named_scope(scope_name):
                return original(self, *args, **kwargs)
        return wrapped

    _SCOPE_LOCK.acquire()
    try:
        for current_type in module_types:
            originals[current_type] = current_type.__call__
            current_type.__call__ = make_wrapped(current_type.__call__)
        yield
    finally:
        for current_type, original in originals.items():
            current_type.__call__ = original
        _SCOPE_LOCK.release()
