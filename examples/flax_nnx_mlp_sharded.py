"""Flax NNX MLP, tensor-parallel sharded with compiler-driven ("Auto") sharding.

This is a "school 1" model: we only annotate how the weights are sharded across a
device mesh and let the XLA compiler insert the communication (an all-reduce). It's
meant for ``trace_model(..., view="per_device")``, which shows the compiler-inserted
collectives; ``view="global"`` shows the same model with no communication.

Uses the explicit NNX state-sharding initialization sequence supported by both
Flax 0.11 and 0.12. Exposes ``model``, ``example_input``, ``trace_context``, and
``views`` for the examples generator.
"""
import jax
import jax.numpy as jnp
from flax import nnx

# This tutorial intentionally simulates an eight-device CPU system.
try:
    jax.config.update("jax_num_cpu_devices", 8)
except RuntimeError:
    pass
cpu_devices = jax.devices("cpu")
if len(cpu_devices) < 8:
    raise RuntimeError("Restart Python and run this example before any JAX operation.")

Auto = jax.sharding.AxisType.Auto
mesh = jax.make_mesh(
    (8,),
    ("model",),
    axis_types=(Auto,),
    devices=cpu_devices,
)


class MLP(nnx.Module):
    def __init__(self, rngs):
        init = nnx.initializers.lecun_normal()
        self.dense0 = nnx.Linear(
            32, 64, use_bias=False, rngs=rngs,
            kernel_init=nnx.with_partitioning(init, (None, "model")),  # column-parallel
        )
        self.dense1 = nnx.Linear(
            64, 32, use_bias=False, rngs=rngs,
            kernel_init=nnx.with_partitioning(init, ("model", None)),  # row-parallel
        )

    def __call__(self, x):
        x = self.dense0(x)
        x = jax.lax.with_sharding_constraint(x, jax.P(None, "model"))
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
    example_input = jax.device_put(jnp.ones((16, 32)), jax.P(None, None))

trace_context = jax.set_mesh(mesh)   # entered around tracing so the mesh is active
views = ("global", "per_device")     # generate both program views for this example
