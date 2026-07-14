"""Flax NNX MLP.

NNX modules emit no named scopes, so passing the module directly to trace_model
lets it inject scopes and render nested containers. Exposes ``model`` (the module)
and ``example_input`` for the examples generator.
"""
import jax.numpy as jnp
from flax import nnx


class MLP(nnx.Module):
    def __init__(self, rngs):
        self.dense0 = nnx.Linear(8, 16, rngs=rngs)
        self.dense1 = nnx.Linear(16, 4, rngs=rngs)

    def __call__(self, x):
        x = self.dense0(x)
        x = nnx.relu(x)
        x = self.dense1(x)
        return x


model = MLP(nnx.Rngs(0))
example_input = jnp.ones((1, 8))
