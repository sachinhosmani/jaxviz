"""Visualize an Equinox MLP's forward pass."""
import jax
import jax.numpy as jnp
import equinox as eqx

import jaxtrace


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


def main():
    model = MLP(jax.random.PRNGKey(0))
    x = jnp.ones((8,))
    jaxtrace.trace_model(lambda x: model(x), x, export_path="equinox_mlp_graph.html")


if __name__ == "__main__":
    main()
