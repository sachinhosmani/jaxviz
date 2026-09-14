import json
from pathlib import Path
import re
from types import SimpleNamespace

from jaxviz.render import _css_size
from scripts.examples_generator import build_display_code, DEFAULT_VIEWS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTED_EXAMPLES = (
    "flax_nnx_mlp_sharded",
    "flax_nnx_mlp_2d_parallel",
    "flax_nnx_head_parallel_attention",
)


def test_css_size_preserves_css_values():
    assert _css_size(800) == "800px"
    assert _css_size("100%") == "100%"


def test_examples_default_to_global_and_per_device_views():
    assert DEFAULT_VIEWS == ("global", "per_device")


def test_build_display_code_removes_metadata_and_adds_trace_call(tmp_path):
    source_path = tmp_path / "example.py"
    source_path.write_text(
        '''"""Example title."""
import jax.numpy as jnp

model = lambda x: x
example_input = jnp.ones((2,))
trace_context = jax.set_mesh(mesh)
views = ("global", "per_device")
'''
    )
    module = SimpleNamespace(example_input=object(), trace_kwargs={"show_constants": True})

    code = build_display_code(source_path, "per_device", module)

    assert '"""Example title."""' not in code
    assert "views =" not in code
    assert "trace_context =" not in code
    assert "from jaxviz import trace_model" in code
    assert "with jax.set_mesh(mesh):" in code
    assert "view='per_device'" in code
    assert "show_constants=True" in code


def test_build_display_code_reuses_existing_trace_context(tmp_path):
    source_path = tmp_path / "example.py"
    source_path.write_text(
        '''import jax
import jax.numpy as jnp

with jax.set_mesh(mesh):
    model = lambda x: x
    example_input = jnp.ones((2,))

trace_context = jax.set_mesh(mesh)
views = ("global", "per_device")
'''
    )
    module = SimpleNamespace(example_input=object())

    code = build_display_code(source_path, "per_device", module)

    assert code.count("with jax.set_mesh(mesh):") == 1
    assert "    trace_model(model, example_input, view='per_device')" in code


def test_distributed_examples_use_idiomatic_model_code():
    sources = {
        name: (PROJECT_ROOT / "examples" / f"{name}.py").read_text()
        for name in DISTRIBUTED_EXAMPLES
    }

    for source in sources.values():
        assert "with_sharding_constraint" not in source
        assert "nnx.get_partition_spec" not in source
        assert "nnx.update" not in source

    attention = sources["flax_nnx_head_parallel_attention"]
    assert "nnx.MultiHeadAttention" in attention
    assert "def attention_weights" not in attention
    assert "def split_heads" not in attention


def test_distributed_website_pages_show_the_communication():
    """Every distributed example must show the collectives the compiler added,
    and must not carry any of the two-shape partitioning data the per-device
    view no longer reports."""
    for name in DISTRIBUTED_EXAMPLES:
        html = (
            PROJECT_ROOT / "docs" / "examples" / f"{name}_per_device.html"
        ).read_text()
        assert "all-reduce" in html
        assert '"node_type": "Collective"' in html
        # the JSON key, not the frontend code that would render it
        assert '"shape_info":' not in html


def test_transformer_example_has_independently_collapsible_blocks():
    html = (
        PROJECT_ROOT / "docs" / "examples" /
        "flax_nnx_transformer_encoder_global.html"
    ).read_text()

    def embedded_json(name):
        match = re.search(rf"^\s*const {name} = (.*);$", html, re.MULTILINE)
        assert match is not None
        return json.loads(match.group(1))

    labels = embedded_json("graph_node_display_names")
    ancestors = embedded_json("ancestor_map")
    modules = embedded_json("parent_module_to_depth")
    block_names = {f"block{index}" for index in range(4)}
    blocks = {
        module for module in modules
        if labels[module] in block_names
    }
    attention_modules = {
        module for module in modules
        if labels[module] == "attention"
    }
    feed_forward_modules = {
        module for module in modules
        if labels[module] == "feed_forward"
    }

    assert len(blocks) == 4
    assert len({ancestors[block] for block in blocks}) == 1
    assert {ancestors[module] for module in attention_modules} == blocks
    assert {ancestors[module] for module in feed_forward_modules} == blocks
    assert "const collapse_modules_after_depth = 1;" in html


def test_every_example_has_global_and_per_device_pages():
    for source_path in sorted((PROJECT_ROOT / "examples").glob("*.py")):
        if source_path.name == "__init__.py":
            continue
        for view in DEFAULT_VIEWS:
            output_path = (
                PROJECT_ROOT / "docs" / "examples" /
                f"{source_path.stem}_{view}.html"
            )
            assert output_path.exists(), output_path.name


def test_every_generated_module_collapse_is_acyclic():
    def embedded_json(html, name):
        match = re.search(rf"^\s*const {name} = (.*);$", html, re.MULTILINE)
        assert match is not None
        return json.loads(match.group(1))

    def descendants(container, ancestors):
        result = set()
        for node in ancestors:
            current = node
            while current in ancestors and ancestors[current] is not None:
                current = ancestors[current]
                if current == container:
                    result.add(node)
                    break
        return result

    def quotient_edges(adjacency, ancestors, collapsed):
        descendant_sets = {
            container: descendants(container, ancestors)
            for container in collapsed
        }
        visible = {
            container
            for container in collapsed
            if not any(
                container in values
                for other, values in descendant_sets.items()
                if other != container
            )
        }

        def representative(node):
            for container in visible:
                if node in descendant_sets[container]:
                    return container
            return node

        edges = set()
        for source, data in adjacency.items():
            for edge in data["edges"]:
                collapsed_source = representative(source)
                collapsed_target = representative(edge["target"])
                if collapsed_source != collapsed_target:
                    edges.add((collapsed_source, collapsed_target))
        return edges

    def assert_acyclic(edges, context):
        outgoing = {}
        for source, target in edges:
            outgoing.setdefault(source, set()).add(target)
            outgoing.setdefault(target, set())
        visiting = set()
        visited = set()

        def visit(node):
            assert node not in visiting, context
            if node in visited:
                return
            visiting.add(node)
            for target in outgoing[node]:
                visit(target)
            visiting.remove(node)
            visited.add(node)

        for node in outgoing:
            visit(node)

    for path in sorted((PROJECT_ROOT / "docs" / "examples").glob("*.html")):
        html = path.read_text()
        adjacency = embedded_json(html, "adj_list")
        ancestors = embedded_json(html, "ancestor_map")
        modules = embedded_json(html, "parent_module_to_depth")
        depth_match = re.search(
            r"^\s*const collapse_modules_after_depth = (\d+);$",
            html,
            re.MULTILINE,
        )
        assert depth_match is not None
        collapse_depth = int(depth_match.group(1))

        def hierarchy_depth(container):
            depth = 0
            current = container
            while ancestors.get(current) is not None:
                depth += 1
                current = ancestors[current]
            return depth

        initial = {
            container for container in modules
            if hierarchy_depth(container) >= collapse_depth
        }
        assert_acyclic(
            quotient_edges(adjacency, ancestors, initial),
            f"initial collapse cycle in {path.name}",
        )
        for container in modules:
            assert_acyclic(
                quotient_edges(adjacency, ancestors, initial | {container}),
                f"collapse cycle in {path.name}: {container}",
            )
