from types import SimpleNamespace

from jaxviz.render import _css_size
from scripts.examples_generator import build_display_code


def test_css_size_preserves_css_values():
    assert _css_size(800) == "800px"
    assert _css_size("100%") == "100%"


def test_build_display_code_removes_metadata_and_adds_trace_call(tmp_path):
    source_path = tmp_path / "example.py"
    source_path.write_text(
        '''"""Example title."""
import jax.numpy as jnp

model = lambda x: x
example_input = jnp.ones((2,))
trace_context = jax.set_mesh(mesh)
levels = ("high", "low")
'''
    )
    module = SimpleNamespace(example_input=object(), trace_kwargs={"show_constants": True})

    code = build_display_code(source_path, "low", module)

    assert '"""Example title."""' not in code
    assert "levels =" not in code
    assert "trace_context =" not in code
    assert "from jaxviz import trace_model" in code
    assert "with jax.set_mesh(mesh):" in code
    assert "level='low'" in code
    assert "show_constants=True" in code
