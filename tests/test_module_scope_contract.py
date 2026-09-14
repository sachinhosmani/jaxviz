import jax
import jax.numpy as jnp
import flax.linen as nn

from jaxviz import _module_scope_context
from jaxviz.jaxpr_to_graph import build_graph


class SharedLinenModule(nn.Module):
    @nn.compact
    def __call__(self, value):
        shared = nn.Dense(8)
        value = shared(value)
        value = nn.relu(value)
        return shared(value)


class CustomMethodModule(nn.Module):
    @nn.compact
    def encode(self, value):
        return nn.Dense(4)(value)


def _trace_global(function, value):
    with _module_scope_context(function):
        closed_jaxpr = jax.make_jaxpr(function)(value)
    return build_graph(closed_jaxpr)


def test_arbitrary_named_scopes_are_not_modules():
    def function(value):
        with jax.named_scope("not_a_module"):
            return jnp.sin(value)

    blobs = _trace_global(function, jnp.ones((4,)))
    assert blobs["module_info"] == {}


def test_vmap_transformations_are_not_modules():
    function = jax.vmap(lambda value: jnp.sin(value))
    blobs = _trace_global(function, jnp.ones((4, 8)))
    assert blobs["module_info"] == {}


def test_reused_linen_module_has_distinct_invocation_containers():
    model = SharedLinenModule()
    value = jnp.ones((2, 8))
    parameters = model.init(jax.random.PRNGKey(0), value)
    blobs = _trace_global(
        lambda current: model.apply(parameters, current),
        value,
    )
    labels = blobs["graph_node_display_names"]
    dense_containers = [
        container for container in blobs["module_info"]
        if labels[container] == "Dense_0"
    ]
    assert len(dense_containers) == 2
    assert dense_containers[0] != dense_containers[1]


def test_linen_custom_method_is_attributed_to_its_module():
    model = CustomMethodModule()
    value = jnp.ones((2, 8))
    parameters = model.init(
        jax.random.PRNGKey(2),
        value,
        method=model.encode,
    )
    blobs = _trace_global(
        lambda current: model.apply(
            parameters,
            current,
            method=model.encode,
        ),
        value,
    )
    labels = {
        blobs["graph_node_display_names"][container]
        for container in blobs["module_info"]
    }
    assert "CustomMethodModule" in labels
    assert "Dense_0" in labels
