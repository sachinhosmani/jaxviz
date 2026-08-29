import jax
import jax.numpy as jnp
import pytest

eqx = pytest.importorskip("equinox")

from jaxviz import _module_scope_context
from jaxviz.jaxpr_to_graph import build_graph


class MLP(eqx.Module):
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, key):
        first, second = jax.random.split(key)
        self.l1 = eqx.nn.Linear(8, 16, key=first)
        self.l2 = eqx.nn.Linear(16, 4, key=second)

    def __call__(self, value):
        value = jax.nn.relu(self.l1(value))
        return self.l2(value)


class SharedMLP(eqx.Module):
    shared: eqx.nn.Linear

    def __init__(self, key):
        self.shared = eqx.nn.Linear(8, 8, key=key)

    def __call__(self, value):
        value = self.shared(value)
        value = jax.nn.relu(value)
        return self.shared(value)


def test_repeated_equinox_module_types_keep_instance_scopes():
    model = MLP(jax.random.PRNGKey(0))
    with _module_scope_context(model):
        closed_jaxpr = jax.make_jaxpr(model)(jnp.ones((8,)))

    blobs = build_graph(closed_jaxpr)
    labels = blobs["graph_node_display_names"]
    containers = {
        labels[container]: container for container in blobs["module_info"]
    }
    assert "l1" in containers
    assert "l2" in containers
    assert "eqx.nn.Linear" not in containers

    ancestors = blobs["ancestor_map"]
    relu = next(node for node in ancestors if node.startswith("relu_"))
    assert labels[ancestors[relu]] == "MLP"


def test_reused_equinox_module_has_distinct_invocation_containers():
    model = SharedMLP(jax.random.PRNGKey(0))
    with _module_scope_context(model):
        closed_jaxpr = jax.make_jaxpr(model)(jnp.ones((8,)))

    blobs = build_graph(closed_jaxpr)
    labels = blobs["graph_node_display_names"]
    shared_containers = [
        container for container in blobs["module_info"]
        if labels[container] == "shared"
    ]
    assert len(shared_containers) == 2
