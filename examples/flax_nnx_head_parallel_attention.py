"""Flax NNX self-attention with data and head parallelism.

The batch is split across the data axis and complete attention heads are split
across the model axis.
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


class HeadParallelAttention(nnx.Module):
    def __init__(self, rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=8,
            in_features=128,
            use_bias=False,
            decode=False,
            kernel_metadata={"out_sharding": (None, "model", None)},
            out_kernel_metadata={"out_sharding": ("model", None, None)},
            rngs=rngs,
        )

    def __call__(self, x):
        return self.attention(x)


@jax.jit
def create_model():
    return HeadParallelAttention(nnx.Rngs(0))


with jax.set_mesh(mesh):
    model = create_model()
    example_input = jax.device_put(
        jnp.ones((8, 32, 128)),
        jax.P("data", None, None),
    )

trace_context = jax.set_mesh(mesh)
views = ("global", "per_device")
trace_kwargs = {"collapse_modules_after_depth": 2}
