"""Generate side-by-side code and visualization pages for JAXViz examples.

Usage:
    python scripts/examples_generator.py
    python scripts/examples_generator.py flax_nnx_mlp_sharded
    python scripts/examples_generator.py flax_nnx_mlp_sharded --view per_device

Each ``examples/*.py`` file defines ``model`` and either ``example_input`` or
``example_args``. Optional metadata controls page titles, trace views, trace
context, and trace keyword arguments. Displayed code is derived from the source
file, so adding an example does not require maintaining a separate snippet.
"""
import argparse
import ast
import contextlib
import html
import importlib.util
from importlib import resources
from pathlib import Path
from string import Template

from jaxviz import trace_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
GENERATED_DIR = EXAMPLES_DIR / "generated"
METADATA_VARS = {
    "code_contents",
    "description",
    "views",
    "title",
    "trace_context",
    "trace_kwargs",
}


def load_example(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assigned_names(node):
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _without_metadata(source):
    tree = ast.parse(source)
    excluded_lines = set()
    context_expression = None

    if (
        tree.body and
        isinstance(tree.body[0], ast.Expr) and
        isinstance(tree.body[0].value, ast.Constant) and
        isinstance(tree.body[0].value.value, str)
    ):
        excluded_lines.update(range(tree.body[0].lineno, tree.body[0].end_lineno + 1))

    for node in tree.body:
        names = _assigned_names(node)
        if "trace_context" in names:
            context_expression = ast.get_source_segment(source, node.value)
        if any(name in METADATA_VARS for name in names):
            excluded_lines.update(range(node.lineno, node.end_lineno + 1))

    lines = [
        line for line_number, line in enumerate(source.splitlines(), start=1)
        if line_number not in excluded_lines
    ]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines), context_expression


def _add_trace_import(source):
    if "from jaxviz import" in source or "import jaxviz" in source:
        return source

    tree = ast.parse(source)
    insert_after = 0
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            insert_after = node.end_lineno
        else:
            break

    lines = source.splitlines()
    lines.insert(insert_after, "from jaxviz import trace_model")
    lines.insert(insert_after + 1, "")
    return "\n".join(lines)


def _insert_into_context(source, context_expression, statement):
    tree = ast.parse(source)
    matching_blocks = [
        node for node in tree.body
        if isinstance(node, ast.With) and any(
            ast.get_source_segment(source, item.context_expr) == context_expression
            for item in node.items
        )
    ]
    if not matching_blocks:
        return None

    block = matching_blocks[-1]
    indentation = " " * block.body[0].col_offset
    lines = source.splitlines()
    lines.insert(block.end_lineno, f"{indentation}{statement}")
    return "\n".join(lines)


def build_display_code(path, view, module):
    override = getattr(module, "code_contents", None)
    if override:
        return override.strip() + "\n"

    source, context_expression = _without_metadata(path.read_text())
    source = _add_trace_import(source).rstrip()
    argument_expression = "*example_args" if hasattr(module, "example_args") else "example_input"
    trace_kwargs = getattr(module, "trace_kwargs", {})
    keyword_expressions = [f"view={view!r}"]
    keyword_expressions.extend(f"{name}={value!r}" for name, value in trace_kwargs.items())
    trace_call = f"trace_model(model, {argument_expression}, {', '.join(keyword_expressions)})"

    if context_expression:
        combined_source = _insert_into_context(source, context_expression, trace_call)
        if combined_source is not None:
            return combined_source + "\n"
        trace_call = f"with {context_expression}:\n    {trace_call}"

    return f"{source}\n\n{trace_call}\n"


def example_title(path, module):
    explicit_title = getattr(module, "title", None)
    if explicit_title:
        return explicit_title
    if module.__doc__:
        first_line = module.__doc__.strip().splitlines()[0].rstrip(".")
        if first_line:
            return first_line
    return path.stem.replace("_", " ").title()


def render_example_page(title, code_contents, graph_html):
    template_text = (
        resources.files("jaxviz.templates")
        .joinpath("example.html")
        .read_text(encoding="utf-8")
    )
    template = Template(template_text)
    return template.safe_substitute({
        "page_title": html.escape(f"{title} · JAXViz"),
        "example_title": html.escape(title),
        "code_contents": html.escape(code_contents),
        "graph_html": graph_html,
    })


def generate_example(path, selected_views=None):
    module = load_example(path)
    model = getattr(module, "model")
    if hasattr(module, "example_args"):
        example_args = tuple(module.example_args)
    else:
        example_args = (getattr(module, "example_input"),)

    views = tuple(getattr(module, "views", ("global",)))
    if selected_views:
        views = tuple(view for view in views if view in selected_views)
    if not views:
        return []

    trace_context = getattr(module, "trace_context", contextlib.nullcontext())
    trace_kwargs = getattr(module, "trace_kwargs", {})
    title = example_title(path, module)
    output_paths = []

    with trace_context:
        for view in views:
            suffix = f"_{view}" if len(getattr(module, "views", ("global",))) > 1 else ""
            output_path = GENERATED_DIR / f"{path.stem}{suffix}.html"
            print(f"Generating {path.stem} ({view})...")
            graph_html = trace_model(
                model,
                *example_args,
                view=view,
                height="100%",
                width="100%",
                return_html=True,
                **trace_kwargs,
            )
            page_html = render_example_page(
                title,
                build_display_code(path, view, module),
                graph_html,
            )
            output_path.write_text(page_html, encoding="utf-8")
            output_paths.append(output_path)
            print(f"Saved output to {output_path}")

    return output_paths


def generate_all(example_names=None, selected_views=None):
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    requested_names = set(example_names or ())
    available_paths = {
        path.stem: path for path in sorted(EXAMPLES_DIR.glob("*.py"))
        if path.name != "__init__.py"
    }
    missing_names = requested_names.difference(available_paths)
    if missing_names:
        missing = ", ".join(sorted(missing_names))
        raise ValueError(f"Unknown examples: {missing}")

    names = sorted(requested_names) if requested_names else sorted(available_paths)
    output_paths = []
    for name in names:
        output_paths.extend(generate_example(available_paths[name], selected_views))
    return output_paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", nargs="*", help="Example file stems to generate")
    parser.add_argument(
        "--view",
        action="append",
        choices=("global", "per_device"),
        dest="views",
        help="Only generate this view; may be passed more than once",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_all(args.examples, set(args.views or ()))
