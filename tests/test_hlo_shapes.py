import jax
import jax.numpy as jnp

from jaxviz._module_scopes import ModuleInvocation, module_scope_name
from jaxviz.enums import NodeType
from jaxviz.hlo_to_graph import (
    _assign_value_module_paths,
    _fallback_global_shape,
    _jaxpr_global_shape_index,
    _module_path_from_op_name,
    _shape_info,
)


def _graph_node(node_type, *targets):
    return {
        "node_type": node_type.value,
        "edges": [{"target": target} for target in targets],
    }


def test_shape_info_identifies_partitioned_dimensions():
    assert _shape_info([3, 8, 512], [3, 32, 1024]) == {
        "global": [3, 32, 1024],
        "local": [3, 8, 512],
        "partitions": [1, 4, 2],
        "axes": None,
        "status": "verified",
    }


def test_shape_info_preserves_unavailable_global_shape():
    assert _shape_info([3, 8, 512], None) == {
        "global": None,
        "local": [3, 8, 512],
        "partitions": None,
        "axes": None,
        "status": "unavailable",
    }


def test_shape_info_rejects_inconsistent_shapes():
    assert _shape_info([3, 7], [3, 32])["status"] == "unavailable"


def test_shape_info_maps_partitioned_dimensions_to_mesh_axes():
    info = _shape_info(
        [3, 8, 512],
        [3, 32, 1024],
        {"data": 4, "model": 2},
    )

    assert info["axes"] == [
        [],
        [{"name": "data", "size": 4}],
        [{"name": "model", "size": 2}],
    ]


def test_shape_info_omits_ambiguous_mesh_axis_mapping():
    info = _shape_info([8, 8], [16, 16], {"x": 2, "y": 2})

    assert info["axes"] is None


def test_jaxpr_fallback_recovers_unique_global_shape():
    assert _fallback_global_shape(
        [8, 16],
        {(16, 64)},
        {"data": 2, "model": 4},
    ) == [16, 64]


def test_jaxpr_fallback_rejects_ambiguous_axis_mapping():
    assert _fallback_global_shape(
        [8, 8],
        {(16, 16)},
        {"x": 2, "y": 2},
    ) is None


def test_jaxpr_fallback_rejects_multiple_matching_shapes():
    assert _fallback_global_shape(
        [8, 16],
        {(16, 64), (32, 32)},
        {"data": 2, "model": 4},
    ) is None


def test_jaxpr_fallback_ignores_incompatible_global_candidates():
    assert _fallback_global_shape(
        [8, 16],
        {(16, 64), (16, 32)},
        {"data": 2, "model": 4},
    ) == [16, 64]


def test_jaxpr_global_shape_index_reads_logical_output_shape():
    def double_width(value):
        return jnp.concatenate((value, value), axis=1)

    closed_jaxpr = jax.make_jaxpr(double_width)(jnp.ones((8, 16)))
    indexed_shapes = {
        shape
        for shapes in _jaxpr_global_shape_index(closed_jaxpr).values()
        for shape in shapes
    }

    assert (8, 32) in indexed_shapes


def test_module_path_reads_deep_tagged_scopes_from_every_provenance():
    first = "/".join([
        "jit(forward)",
        module_scope_name("Model", 1),
        module_scope_name("block", 1),
        module_scope_name("attention", 1),
        "vmap()",
        "transpose",
    ])
    second = "/".join([
        "jit(forward)",
        module_scope_name("Model", 1),
        module_scope_name("block", 1),
        module_scope_name("attention", 1),
        "reshape",
    ])

    assert _module_path_from_op_name(first + ";" + second) == [
        ModuleInvocation("Model", 1),
        ModuleInvocation("block", 1),
        ModuleInvocation("attention", 1),
    ]


def test_module_path_keeps_only_unambiguous_common_ancestry():
    op_name = (
        f"jit(forward)/{module_scope_name('Model', 1)}/"
        f"{module_scope_name('left', 1)}/add;"
        f"jit(forward)/{module_scope_name('Model', 1)}/"
        f"{module_scope_name('right', 1)}/add"
    )

    assert _module_path_from_op_name(op_name) == [ModuleInvocation("Model", 1)]


def test_module_path_rejects_partially_unattributed_provenance():
    op_name = (
        f"jit(forward)/{module_scope_name('Model', 1)}/"
        f"{module_scope_name('attention', 1)}/transpose;"
        "jit(forward)/transpose"
    )

    assert _module_path_from_op_name(op_name) == []


def test_value_nodes_follow_their_only_consuming_module():
    model_path = [ModuleInvocation("MLP", 1)]
    adj_list = {
        "constant": _graph_node(NodeType.CONSTANT, "maximum"),
        "maximum": _graph_node(NodeType.OPERATION),
    }
    node_modpath = {"constant": [], "maximum": model_path}

    _assign_value_module_paths(adj_list, node_modpath)

    assert node_modpath["constant"] == model_path


def test_shared_values_use_their_consumers_common_module():
    model = ModuleInvocation("Model", 1)
    adj_list = {
        "parameter": _graph_node(NodeType.PARAMETER, "left_op", "right_op"),
        "left_op": _graph_node(NodeType.OPERATION),
        "right_op": _graph_node(NodeType.OPERATION),
    }
    node_modpath = {
        "parameter": [],
        "left_op": [model, ModuleInvocation("left", 1)],
        "right_op": [model, ModuleInvocation("right", 1)],
    }

    _assign_value_module_paths(adj_list, node_modpath)

    assert node_modpath["parameter"] == [model]


def test_value_nodes_stay_at_root_when_any_consumer_is_at_root():
    adj_list = {
        "constant": _graph_node(NodeType.CONSTANT, "module_op", "root_op"),
        "module_op": _graph_node(NodeType.OPERATION),
        "root_op": _graph_node(NodeType.OPERATION),
    }
    node_modpath = {
        "constant": [],
        "module_op": [ModuleInvocation("MLP", 1)],
        "root_op": [],
    }

    _assign_value_module_paths(adj_list, node_modpath)

    assert node_modpath["constant"] == []
