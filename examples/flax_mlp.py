"""Flax Linen MLP.

Exposes ``model`` (the callable to trace) and ``example_input`` for the examples
generator.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(4)(x)
        return x


_mlp = MLP()
example_input = jnp.ones((1, 8))
_params = _mlp.init(jax.random.PRNGKey(0), example_input)


def model(x):
    return _mlp.apply(_params, x)
