import pytest

from jaxviz.hierarchy import validate_collapsible_hierarchy


def test_non_convex_module_hierarchy_is_rejected():
    adjacency = {
        "inside_before": {
            "edges": [{"target": "outside"}],
            "node_type": "Operation",
        },
        "outside": {
            "edges": [{"target": "inside_after"}],
            "node_type": "Operation",
        },
        "inside_after": {"edges": [], "node_type": "Operation"},
    }
    ancestors = {
        "inside_before": "module",
        "outside": None,
        "inside_after": "module",
        "module": None,
    }

    with pytest.raises(ValueError, match="not graph-convex"):
        validate_collapsible_hierarchy(
            adjacency,
            ancestors,
            {"module": {"name": "shared"}},
        )
