"""Flax NNX MLP with data and tensor parallelism on a 2D device mesh.

The batch is split across the data axis while the hidden features are split
across the model axis. This makes the hidden activation use both mesh axes at
once and lets the per-device graph reveal the model-parallel communication.
"""
import jax
import jax.numpy as jnp
from flax import nnx

# This tutorial intentionally simulates an eight-device CPU system. Configure
# devices before the first JAX operation; production code should use its real
# accelerator devices instead.
try:
    jax.config.update("jax_num_cpu_devices", 8)
except RuntimeError:
    pass
cpu_devices = jax.devices("cpu")
if len(cpu_devices) < 8:
    raise RuntimeError("Restart Python and run this example before any JAX operation.")

Auto = jax.sharding.AxisType.Auto
mesh = jax.make_mesh(
    (2, 4),
    ("data", "model"),
    axis_types=(Auto, Auto),
    devices=cpu_devices,
)


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
        x = jax.lax.with_sharding_constraint(x, jax.P("data", "model"))
        x = nnx.relu(x)
        x = self.dense1(x)
        return x


@nnx.jit
def create_sharded_model():
    model = MLP(nnx.Rngs(0))
    state = nnx.state(model)
    partition_specs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, partition_specs)
    nnx.update(model, sharded_state)
    return model


with jax.set_mesh(mesh):
    model = create_sharded_model()
    # Global (16, 32) becomes local (8, 32) on each data replica.
    example_input = jax.device_put(jnp.ones((16, 32)), jax.P("data", None))

trace_context = jax.set_mesh(mesh)
views = ("global", "per_device")
trace_kwargs = {"collapse_modules_after_depth": 2}
