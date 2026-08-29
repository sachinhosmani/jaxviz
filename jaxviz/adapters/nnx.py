from flax import nnx

from ._object import object_module_scopes


def named_scopes(model, root_name=None):
    return object_module_scopes(model, nnx.Module, root_name=root_name)
