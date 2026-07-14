"""Visualize a Flax NNX MLP's forward pass.

NNX does not emit named scopes, so the graph is flat by default. The nnx adapter
injects scopes so that submodules render as nested containers.
"""
import jax.numpy as jnp
from flax import nnx

import jaxtrace
from jaxtrace.adapters.nnx import named_scopes


class MLP(nnx.Module):
    def __init__(self, rngs):
        self.dense0 = nnx.Linear(8, 16, rngs=rngs)
        self.dense1 = nnx.Linear(16, 4, rngs=rngs)

    def __call__(self, x):
        x = self.dense0(x)
        x = nnx.relu(x)
        x = self.dense1(x)
        return x


def main():
    x = jnp.ones((1, 8))
    model = MLP(nnx.Rngs(0))
    with named_scopes(model):
        jaxtrace.trace_model(lambda x: model(x), x,
                             export_path="flax_nnx_mlp_graph.html")


if __name__ == "__main__":
    main()
