"""Flax NNX self-attention with data and head parallelism.

The batch is split across the data axis and complete attention heads are split
across the model axis. The output projection then combines the per-head partial
results across model shards.
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
    (2, 4),
    ("data", "model"),
    axis_types=(Auto, Auto),
    devices=cpu_devices,
)


@jax.jit
def attention_weights(query, key):
    scale = jnp.sqrt(jnp.asarray(query.shape[-1], dtype=query.dtype))
    scores = jnp.einsum("bhsd,bhtd->bhst", query, key) / scale
    return jax.nn.softmax(scores, axis=-1)


class HeadParallelAttention(nnx.Module):
    def __init__(self, rngs):
        self.num_heads = 8
        self.head_dim = 16
        embedding_dim = self.num_heads * self.head_dim
        init = nnx.initializers.lecun_normal()

        # Column-parallel projections assign complete heads to model shards.
        projection = nnx.with_partitioning(init, (None, "model"))
        self.query = nnx.Linear(
            embedding_dim,
            embedding_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=projection,
        )
        self.key = nnx.Linear(
            embedding_dim,
            embedding_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=projection,
        )
        self.value = nnx.Linear(
            embedding_dim,
            embedding_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=projection,
        )
        # The row-parallel output projection combines the head shards.
        self.output = nnx.Linear(
            embedding_dim,
            embedding_dim,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.with_partitioning(init, ("model", None)),
        )

    def split_heads(self, x):
        batch, sequence, _ = x.shape
        x = x.reshape(batch, sequence, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def __call__(self, x):
        projection_sharding = jax.P("data", None, "model")
        head_sharding = jax.P("data", "model", None, None)

        query = jax.lax.with_sharding_constraint(self.query(x), projection_sharding)
        key = jax.lax.with_sharding_constraint(self.key(x), projection_sharding)
        value = jax.lax.with_sharding_constraint(self.value(x), projection_sharding)

        query = jax.lax.with_sharding_constraint(self.split_heads(query), head_sharding)
        key = jax.lax.with_sharding_constraint(self.split_heads(key), head_sharding)
        value = jax.lax.with_sharding_constraint(self.split_heads(value), head_sharding)

        probabilities = attention_weights(query, key)
        probabilities = jax.lax.with_sharding_constraint(probabilities, head_sharding)
        context = jnp.einsum("bhst,bhtd->bhsd", probabilities, value)
        context = jax.lax.with_sharding_constraint(context, head_sharding)

        context = context.transpose(0, 2, 1, 3)
        context = context.reshape(x.shape)
        context = jax.lax.with_sharding_constraint(context, projection_sharding)
        # Expand this module in the per-device graph to see its all-reduce.
        return self.output(context)


@nnx.jit
def create_sharded_model():
    model = HeadParallelAttention(nnx.Rngs(0))
    state = nnx.state(model)
    partition_specs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, partition_specs)
    nnx.update(model, sharded_state)
    return model


with jax.set_mesh(mesh):
    model = create_sharded_model()
    # Global (8, 32, 128) becomes local (4, 32, 128) across data replicas.
    example_input = jax.device_put(
        jnp.ones((8, 32, 128)),
        jax.P("data", None, None),
    )

trace_context = jax.set_mesh(mesh)
views = ("global", "per_device")
trace_kwargs = {"collapse_modules_after_depth": 1}
