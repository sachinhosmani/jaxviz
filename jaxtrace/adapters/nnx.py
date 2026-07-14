"""Nesting adapter for Flax NNX.

NNX modules are plain objects (attributes and method calls) and do not emit
``jax.named_scope``s, so a jaxpr traced from an NNX model has an empty name_stack
and renders flat.

``named_scopes`` restores nesting: it walks the module tree and temporarily wraps
each module type's ``__call__`` so that calling a submodule pushes a
``jax.named_scope`` named after the attribute it was assigned to. Python resolves
``obj(...)`` via the type's ``__call__``, so the wrap is applied to the type but
branches on ``id(self)`` to give each instance its own name; scopes then nest
naturally as calls nest.

    from jaxtrace.adapters.nnx import named_scopes
    with named_scopes(model):
        jaxtrace.trace_model(lambda x: model(x), x)

NNX caches its traced call per model instance, so trace a fresh model instance
inside the context rather than one that has already been traced without scopes.
"""
import contextlib

import jax
from flax import nnx


def _collect(module, id_to_name, modules):
    """Map id(submodule) -> attribute name, and gather every module instance."""
    modules.append(module)
    for attr, val in vars(module).items():
        if isinstance(val, nnx.Module) and id(val) not in id_to_name:
            id_to_name[id(val)] = attr
            _collect(val, id_to_name, modules)


@contextlib.contextmanager
def named_scopes(model, root_name=None):
    id_to_name = {id(model): root_name or type(model).__name__}
    modules = []
    _collect(model, id_to_name, modules)

    types = {type(m) for m in modules if "__call__" in vars(type(m))}
    originals = {}

    def make_wrapped(orig):
        def wrapped(self, *args, **kwargs):
            name = id_to_name.get(id(self))
            if name is None:
                return orig(self, *args, **kwargs)
            with jax.named_scope(name):
                return orig(self, *args, **kwargs)
        return wrapped

    try:
        for cls in types:
            originals[cls] = cls.__call__
            cls.__call__ = make_wrapped(cls.__call__)
        yield
    finally:
        for cls, orig in originals.items():
            cls.__call__ = orig
