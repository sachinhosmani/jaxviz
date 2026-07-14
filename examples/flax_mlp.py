"""Visualize a Flax Linen MLP's forward pass."""
import jax
import jax.numpy as jnp
import flax.linen as nn

import jaxtrace


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        x = nn.Dense(4)(x)
        return x


def main():
    model = MLP()
    x = jnp.ones((1, 8))
    params = model.init(jax.random.PRNGKey(0), x)

    jaxtrace.trace_model(lambda x: model.apply(params, x), x,
                         export_path="flax_mlp_graph.html")


if __name__ == "__main__":
    main()
