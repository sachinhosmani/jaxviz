"""Flax NNX Transformer encoder with independently collapsible blocks.

Four pre-norm encoder blocks expose repeated attention and feed-forward module
hierarchies, making it possible to expand one block while leaving the others
collapsed.
"""
import jax.numpy as jnp
from flax import nnx


class FeedForward(nnx.Module):
    def __init__(self, features, hidden_features, *, rngs):
        self.up_projection = nnx.Linear(features, hidden_features, rngs=rngs)
        self.down_projection = nnx.Linear(hidden_features, features, rngs=rngs)

    def __call__(self, x):
        x = self.up_projection(x)
        x = nnx.gelu(x)
        return self.down_projection(x)


class TransformerBlock(nnx.Module):
    def __init__(self, features, num_heads, mlp_features, *, rngs):
        self.norm0 = nnx.LayerNorm(features, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=features,
            decode=False,
            rngs=rngs,
        )
        self.norm1 = nnx.LayerNorm(features, rngs=rngs)
        self.feed_forward = FeedForward(features, mlp_features, rngs=rngs)

    def __call__(self, x):
        x = x + self.attention(self.norm0(x))
        x = x + self.feed_forward(self.norm1(x))
        return x


class TransformerEncoder(nnx.Module):
    def __init__(self, *, rngs):
        self.block0 = TransformerBlock(64, 4, 128, rngs=rngs)
        self.block1 = TransformerBlock(64, 4, 128, rngs=rngs)
        self.block2 = TransformerBlock(64, 4, 128, rngs=rngs)
        self.block3 = TransformerBlock(64, 4, 128, rngs=rngs)
        self.final_norm = nnx.LayerNorm(64, rngs=rngs)

    def __call__(self, x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.final_norm(x)


model = TransformerEncoder(rngs=nnx.Rngs(0))
example_input = jnp.ones((2, 16, 64))
trace_kwargs = {"collapse_modules_after_depth": 1}
