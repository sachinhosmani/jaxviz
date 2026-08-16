"""Generate HTML visualizations for all example models.

Usage:
    python scripts/examples_generator.py

Each file in examples/ exposes ``model`` (the callable to trace) and
``example_input``, and may optionally expose ``trace_context`` (a context manager
entered around tracing) and ``levels``. This script imports each one and writes
its graph to examples/generated/<name>.html, or <name>_<level>.html when more
than one level is requested.
"""
import contextlib
import importlib.util
from pathlib import Path

from jaxviz import trace_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
GENERATED_DIR = EXAMPLES_DIR / "generated"


def load_example(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_all():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue

        module = load_example(path)
        model = getattr(module, "model")
        example_input = getattr(module, "example_input")
        trace_context = getattr(module, "trace_context", contextlib.nullcontext())
        levels = getattr(module, "levels", ("high",))

        with trace_context:
            for level in levels:
                suffix = f"_{level}" if len(levels) > 1 else ""
                output_path = GENERATED_DIR / f"{path.stem}{suffix}.html"
                print(f"Generating {path.stem} ({level})...")
                trace_model(
                    model,
                    example_input,
                    level=level,
                    export_format="html",
                    export_path=str(output_path),
                    height=805,
                    width="100%",
                )
                print(f"Saved output to {output_path}")


if __name__ == "__main__":
    generate_all()
