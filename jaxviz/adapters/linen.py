import contextlib
from collections import defaultdict

import flax.linen as nn
import jax

from .._module_scopes import module_scope_name


@contextlib.contextmanager
def named_scopes():
    invocation_counts = defaultdict(int)
    active_modules = []

    def interceptor(next_function, args, kwargs, context):
        module = context.module
        path = tuple(getattr(module, "path", ()) or ())
        name = path[-1] if path else type(module).__name__
        identity = path or (type(module).__name__,)
        if identity in active_modules:
            return next_function(*args, **kwargs)
        invocation_counts[identity] += 1
        scope_name = module_scope_name(name, invocation_counts[identity])
        active_modules.append(identity)
        try:
            with jax.named_scope(scope_name):
                return next_function(*args, **kwargs)
        finally:
            active_modules.pop()

    with nn.intercept_methods(interceptor):
        yield
