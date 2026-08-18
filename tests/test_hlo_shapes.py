from jaxviz.hlo_to_graph import _shape_info


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
