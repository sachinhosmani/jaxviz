import jax
import jax.numpy as jnp
import flax.linen as nn

from jaxviz import _module_scope_context
from jaxviz.jaxpr_to_graph import build_graph


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        return nn.Dense(4)(x)


def _mlp_blobs():
    model = MLP()
    x = jnp.ones((1, 8))
    params = model.init(jax.random.PRNGKey(0), x)
    forward = lambda value: model.apply(params, value)
    with _module_scope_context(forward):
        closed_jaxpr = jax.make_jaxpr(forward)(x)
    return build_graph(closed_jaxpr)


def test_has_input_and_output_nodes():
    node_types = {d["node_type"] for d in _mlp_blobs()["adj_list"].values()}
    assert "Input" in node_types
    assert "Output" in node_types


def test_edges_carry_shapes():
    for data in _mlp_blobs()["adj_list"].values():
        for edge in data["edges"]:
            assert edge["dims"].startswith("(")


def test_container_ids_are_dot_safe():
    # Ids reach a DOT renderer, so they must not contain '/' or '.'.
    for node in _mlp_blobs()["ancestor_map"]:
        assert "/" not in node and "." not in node


def test_dense_ops_nest_under_a_top_level_module():
    blobs = _mlp_blobs()
    ancestor_map = blobs["ancestor_map"]
    roots = {parent for parent in ancestor_map.values() if parent not in ancestor_map}
    assert any(
        blobs["graph_node_display_names"].get(root) == "MLP"
        for root in roots
    )


def test_flat_graph_when_no_named_scopes():
    x = jnp.ones((4,))
    blobs = build_graph(jax.make_jaxpr(lambda x: jnp.sum(x * 2))(x))
    assert blobs["ancestor_map"] == {}


def test_constants_hidden_by_default_shown_when_enabled():
    # x * 2 has a literal operand: no Constant node by default, one labeled "2.0"
    # (feeding the multiply) when show_constants=True.
    closed = jax.make_jaxpr(lambda x: jnp.sum(x * 2))(jnp.ones((4,)))

    off = build_graph(closed, show_constants=False)
    assert not any(d["node_type"] == "Constant" for d in off["adj_list"].values())

    on = build_graph(closed, show_constants=True)
    const_nodes = [n for n, d in on["adj_list"].items() if d["node_type"] == "Constant"]
    assert const_nodes
    # Labelled generically; the actual value lives in func_info for the click popup.
    assert on["graph_node_display_names"][const_nodes[0]] == "scalar"
    assert 2.0 in on["func_info"][const_nodes[0]]["values"]


def test_reused_tensor_shares_edge_data_id():
    # A tensor consumed by several ops yields edges with the same edge_data_id, so a
    # collapsed container can merge them into a single edge (distinct targets remain
    # separate when expanded).
    def f(x):
        y = jnp.sin(x)
        return jnp.cos(y) + jnp.exp(y)

    adj = build_graph(jax.make_jaxpr(f)(jnp.ones((4,))))["adj_list"]
    sin_nodes = [n for n in adj if n.startswith("sin_")]
    assert sin_nodes
    edges = adj[sin_nodes[0]]["edges"]
    assert len(edges) == 2
    assert edges[0]["edge_data_id"] == edges[1]["edge_data_id"]
    assert edges[0]["target"] != edges[1]["target"]
