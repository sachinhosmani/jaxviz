import jax
import jax.numpy as jnp

from jaxviz.hlo_to_graph import (
    _fallback_global_shape,
    _jaxpr_global_shape_index,
    _shape_info,
)


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
