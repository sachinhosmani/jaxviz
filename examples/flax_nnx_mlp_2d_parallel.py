"""Flax NNX MLP with data and tensor parallelism on a 2D mesh.

The batch is split across the data axis while each linear layer is tensor
parallel across the model axis.
"""
import jax

# Simulate eight devices so the example runs on a CPU-only machine.
jax.config.update("jax_num_cpu_devices", 8)

import jax.numpy as jnp
from flax import nnx


Auto = jax.sharding.AxisType.Auto
mesh = jax.make_mesh(
    (2, 4),
    ("data", "model"),
    axis_types=(Auto, Auto),
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
    example_input = jax.device_put(jnp.ones((16, 32)), jax.P("data", None))

trace_context = jax.set_mesh(mesh)
views = ("global", "per_device")
trace_kwargs = {"collapse_modules_after_depth": 2}
