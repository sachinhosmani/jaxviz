"""Contract for the per-device view.

The view draws the final compiled module -- the code that actually runs. Every
edge carries the shape its instruction states, labelled the same way the traced
view labels its own edges. Nothing is recovered from an earlier compilation
stage and no module hierarchy is reconstructed.
"""
import re

import jax
import jax.numpy as jnp
import pytest

from jaxviz.enums import NodeType
from jaxviz.hlo_to_graph import (
    _hlo_shape_to_dims,
    _label_for,
    _parse_module,
    _parse_operands,
    build_hlo_graph,
)


NEWER_SHARDING_API = all(
    hasattr(jax, name) for name in ("make_mesh", "set_mesh", "P")
) and hasattr(jax.sharding, "AxisType")


# --------------------------------------------------------------------------
# Edge labels are just the shape, as in the traced view
# --------------------------------------------------------------------------
def test_edge_label_is_the_stated_shape():
    assert _hlo_shape_to_dims("f32[16,8]{1,0}") == "(16, 8)"
    assert _hlo_shape_to_dims("f32[]") == "( )"


# --------------------------------------------------------------------------
# Both HLO text dialects parse: names with and without a leading '%'
# --------------------------------------------------------------------------
def test_module_parses_names_written_without_a_percent_prefix():
    text = (
        "HloModule m\n"
        "\n"
        "ENTRY main.5 {\n"
        "  Arg_0.1 = f32[8,16]{1,0} parameter(0)\n"
        "  ROOT tanh.4 = f32[8,16]{1,0} tanh(Arg_0.1)\n"
        "}\n"
    )
    computations, entry = _parse_module(text)
    assert entry.name == "main.5"
    assert [i.opcode for i in entry.instructions] == ["parameter", "tanh"]
    assert entry.instructions[1].operands == ["Arg_0.1"]


def test_module_parses_names_written_with_a_percent_prefix():
    text = (
        "HloModule m\n"
        "\n"
        "%fused_computation (p: f32[8]) -> f32[8] {\n"
        "  %p = f32[8]{0} parameter(0)\n"
        "  ROOT %t = f32[8]{0} tanh(%p)\n"
        "}\n"
        "\n"
        "ENTRY %main (a: f32[8]) -> f32[8] {\n"
        "  %a = f32[8]{0} parameter(0)\n"
        "  ROOT %f = f32[8]{0} fusion(%a), kind=kLoop, calls=%fused_computation\n"
        "}\n"
    )
    computations, entry = _parse_module(text)
    assert entry.name == "main"
    assert "fused_computation" in computations
    assert entry.instructions[1].calls == ["fused_computation"]


def test_a_computation_closing_the_file_is_still_recorded():
    text = "HloModule m\n\nENTRY main {\n  ROOT %a = f32[] constant(1)\n}"
    _, entry = _parse_module(text)
    assert entry.name == "main"


def test_literal_arguments_are_not_operands():
    assert _parse_operands("0") == []
    assert _parse_operands("%a, %b") == ["a", "b"]
    assert _parse_operands("a, b") == ["a", "b"]


def test_a_fusion_is_named_for_the_operation_it_was_built_from():
    computations, entry = _parse_module(
        "HloModule m\n"
        "\n"
        "ENTRY main {\n"
        '  ROOT %f = f32[8]{0} fusion(%a), calls=%c, metadata={op_name="jit(m)/dot_general[x=1]"}\n'
        "}\n"
    )
    assert _label_for(entry.instructions[0]) == "dot_general"


# --------------------------------------------------------------------------
# End to end: nothing is shown that the compiled module does not state
# --------------------------------------------------------------------------
@pytest.mark.skipif(not NEWER_SHARDING_API,
                    reason="needs the explicit-sharding APIs")
def test_compiled_view_shows_collectives_and_only_stated_global_shapes():
    if len(jax.devices()) < 4:
        pytest.skip("needs at least four devices")

    explicit = jax.sharding.AxisType.Explicit
    mesh = jax.make_mesh((1, 4), ("data", "model"),
                         axis_types=(explicit, explicit))

    def model(x, w1, w2):
        hidden = jnp.tanh(jnp.dot(x, w1))
        return jnp.dot(hidden, w2, out_sharding=jax.P("data", None))

    with jax.set_mesh(mesh):
        x = jax.device_put(jnp.ones((8, 16)), jax.P("data", None))
        w1 = jax.device_put(jnp.ones((16, 32)), jax.P(None, "model"))
        w2 = jax.device_put(jnp.ones((32, 16)), jax.P("model", None))
        lowered = jax.jit(model).lower(x, w1, w2)
        blobs = build_hlo_graph(lowered, mesh_shape=dict(mesh.shape))

    adjacency = blobs["adj_list"]

    collectives = [
        node for node, data in adjacency.items()
        if data["node_type"] == NodeType.COLLECTIVE.value
    ]
    assert collectives, "the row-parallel multiply needs an all-reduce"

    # No module hierarchy, and every edge carries a plain shape label.
    assert blobs["module_info"] == {} or all(
        info.get("type") for info in blobs["module_info"].values()
    )
    for data in adjacency.values():
        for edge in data["edges"]:
            assert "shape_info" not in edge
            assert re.fullmatch(r"\(.*\)", edge["dims"])
