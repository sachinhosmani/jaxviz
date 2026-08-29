import jax
import jax.numpy as jnp
import pytest

nnx = pytest.importorskip("flax.nnx")

from jaxviz import _module_scope_context
from jaxviz.jaxpr_to_graph import build_graph


class SharedNNXModule(nnx.Module):
    def __init__(self, rngs):
        self.shared = nnx.Linear(8, 8, rngs=rngs)

    def __call__(self, value):
        value = self.shared(value)
        value = nnx.relu(value)
        return self.shared(value)


def test_reused_nnx_module_has_distinct_invocation_containers():
    model = SharedNNXModule(nnx.Rngs(0))
    with _module_scope_context(model):
        closed_jaxpr = jax.make_jaxpr(model)(jnp.ones((2, 8)))
    blobs = build_graph(closed_jaxpr)
    labels = blobs["graph_node_display_names"]
    shared_containers = [
        container for container in blobs["module_info"]
        if labels[container] == "shared"
    ]
    assert len(shared_containers) == 2
    assert shared_containers[0] != shared_containers[1]
