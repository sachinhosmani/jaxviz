"""Flax NNX MLP.

NNX emits no named scopes, so ``trace_context`` wraps tracing in ``named_scopes``
to produce a nested graph. Exposes ``model``, ``example_input`` and
``trace_context`` for the examples generator.
"""
import jax.numpy as jnp
from flax import nnx

from jaxviz.adapters.nnx import named_scopes


class MLP(nnx.Module):
    def __init__(self, rngs):
        self.dense0 = nnx.Linear(8, 16, rngs=rngs)
        self.dense1 = nnx.Linear(16, 4, rngs=rngs)

    def __call__(self, x):
        x = self.dense0(x)
        x = nnx.relu(x)
        x = self.dense1(x)
        return x


_mlp = MLP(nnx.Rngs(0))
example_input = jnp.ones((1, 8))
trace_context = named_scopes(_mlp)


def model(x):
    return _mlp(x)
