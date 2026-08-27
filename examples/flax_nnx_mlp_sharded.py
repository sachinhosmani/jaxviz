"""Flax NNX MLP with one-axis tensor parallelism.

The first linear layer shards its output features across the model axis. The
second linear layer consumes those feature shards and JAX inserts the reduction
needed to produce a replicated output.
"""
import jax

# Simulate eight devices so the example runs on a CPU-only machine.
jax.config.update("jax_num_cpu_devices", 8)

import jax.numpy as jnp
from flax import nnx


Auto = jax.sharding.AxisType.Auto
mesh = jax.make_mesh(
    (8,),
    ("model",),
    axis_types=(Auto,),
    devices=jax.devices("cpu"),
)


class MLP(nnx.Module):
    def __init__(self, rngs):
        init = nnx.initializers.lecun_normal()
        self.dense0 = nnx.Linear(
            32,
            64,
            use_bias=False,
            kernel_init=nnx.with_partitioning(init, (None, "model")),
            rngs=rngs,
        )
        self.dense1 = nnx.Linear(
            64,
            32,
            use_bias=False,
            kernel_init=nnx.with_partitioning(init, ("model", None)),
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.dense0(x)
        x = nnx.relu(x)
        return self.dense1(x)


@jax.jit
def create_model():
    return MLP(nnx.Rngs(0))


with jax.set_mesh(mesh):
    model = create_model()
    example_input = jax.device_put(jnp.ones((16, 32)), jax.P(None, None))

trace_context = jax.set_mesh(mesh)
views = ("global", "per_device")
