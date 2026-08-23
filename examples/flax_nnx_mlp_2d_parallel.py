"""Flax NNX MLP with data and tensor parallelism on a 2D device mesh.

The batch is split across the data axis while the hidden features are split
across the model axis. This makes the hidden activation use both mesh axes at
once and lets the per-device graph reveal the model-parallel communication.
"""
import jax
import jax.numpy as jnp
from flax import nnx

Auto = jax.sharding.AxisType.Auto
mesh = jax.make_mesh(
    (2, 4),
    ("data", "model"),
    axis_types=(Auto, Auto),
)
nnx.use_eager_sharding(True)


class MLP(nnx.Module):
    def __init__(self, rngs):
        init = nnx.initializers.lecun_normal()
        self.dense0 = nnx.Linear(
            32,
            64,
            use_bias=False,
            rngs=rngs,
            # Each data replica owns a column-parallel slice of this weight.
            kernel_init=nnx.with_partitioning(init, (None, "model")),
        )
        self.dense1 = nnx.Linear(
            64,
            32,
            use_bias=False,
            rngs=rngs,
            # Row-parallel slices are combined across the model axis.
            kernel_init=nnx.with_partitioning(init, ("model", None)),
        )

    def __call__(self, x):
        x = self.dense0(x)
        x = nnx.relu(x)
        x = self.dense1(x)
        return x


with jax.set_mesh(mesh):
    model = MLP(nnx.Rngs(0))
    # Global (16, 32) becomes local (8, 32) on each data replica.
    example_input = jax.device_put(jnp.ones((16, 32)), jax.P("data", None))

trace_context = jax.set_mesh(mesh)
views = ("global", "per_device")
trace_kwargs = {"collapse_modules_after_depth": 2}
