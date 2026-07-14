"""Equinox MLP.

Exposes ``model`` (the callable to trace) and ``example_input`` for the examples
generator.
"""
import jax
import jax.numpy as jnp
import equinox as eqx


class MLP(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.l1 = eqx.nn.Linear(8, 16, key=k1)
        self.l2 = eqx.nn.Linear(16, 4, key=k2)

    def __call__(self, x):
        x = jax.nn.relu(self.l1(x))
        return self.l2(x)


_mlp = MLP(jax.random.PRNGKey(0))
example_input = jnp.ones((8,))


def model(x):
    return _mlp(x)
