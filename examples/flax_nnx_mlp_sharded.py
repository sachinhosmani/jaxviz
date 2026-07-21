"""Flax NNX MLP, tensor-parallel sharded with compiler-driven ("Auto") sharding.

This is a "school 1" model: we only annotate how the weights are sharded across a
device mesh and let the XLA compiler insert the communication (an all-reduce). It's
meant for ``trace_model(..., level="low")``, which shows the compiler-inserted
collectives; ``level="high"`` shows the same model with no communication.

Requires a recent jax/flax (make_mesh, AxisType.Auto, nnx.use_eager_sharding) and a
mesh in context — enter ``trace_context`` around the trace. Exposes ``model``,
``example_input``, ``trace_context``, and ``levels`` for the examples generator.
"""
import jax
import jax.numpy as jnp
from flax import nnx

Auto = jax.sharding.AxisType.Auto
mesh = jax.make_mesh((8,), ("model",), axis_types=(Auto,))
nnx.use_eager_sharding(True)


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
        x = nnx.relu(x)
        x = self.dense1(x)
        return x


with jax.set_mesh(mesh):
    model = MLP(nnx.Rngs(0))

example_input = jnp.ones((16, 32))
trace_context = jax.set_mesh(mesh)   # entered around tracing so the mesh is active
levels = ("high", "low")             # generate both graph types for this example
