"""Flax NNX self-attention with data and head parallelism.

The batch is split across the data axis and complete attention heads are split
across the model axis. The output projection then combines the per-head partial
results across model shards.
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
        query = self.split_heads(self.query(x))
        key = self.split_heads(self.key(x))
        value = self.split_heads(self.value(x))

        probabilities = attention_weights(query, key)
        context = jnp.einsum("bhst,bhtd->bhsd", probabilities, value)

        context = context.transpose(0, 2, 1, 3)
        context = context.reshape(x.shape)
        # Expand this module in the low-level graph to see its all-reduce.
        return self.output(context)


with jax.set_mesh(mesh):
    model = HeadParallelAttention(nnx.Rngs(0))
    # Global (8, 32, 128) becomes local (4, 32, 128) across data replicas.
    example_input = jax.device_put(
        jnp.ones((8, 32, 128)),
        jax.P("data", None, None),
    )

trace_context = jax.set_mesh(mesh)
levels = ("high", "low")
trace_kwargs = {"collapse_modules_after_depth": 1}
