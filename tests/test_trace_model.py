import jax.numpy as jnp
import pytest

from jaxviz import trace_model


def test_trace_model_rejects_unknown_view():
    with pytest.raises(ValueError, match="Invalid view"):
        trace_model(lambda x: x, jnp.ones((2,)), view="high")
