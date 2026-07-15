"""jaxviz: interactive forward-pass visualization for JAX models.

The computation graph is obtained from ``jax.make_jaxpr`` and converted into the
data structures rendered by the bundled interactive frontend.

    import jaxviz
    jaxviz.trace_model(fn, x)   # fn: a traceable callable, x: example input(s)
"""
import contextlib

import jax

from .enums import ExportFormat
from .jaxpr_to_graph import build_graph
from .render import plot_graph, validate_export_format

__all__ = ["trace_model", "ExportFormat"]


def _module_scope_context(fn):
    """Return a context manager that adds module nesting for object-based models
    that do not emit named scopes on their own.

    Flax NNX modules are plain objects and produce a flat jaxpr; passing the module
    directly lets us walk it and inject ``jax.named_scope``s so submodules render as
    nested containers. Anything else (Flax Linen closures, raw functions) is traced
    unchanged.
    """
    try:
        from flax import nnx
    except ImportError:
        return contextlib.nullcontext()

    if isinstance(fn, nnx.Module):
        from .adapters.nnx import named_scopes
        return named_scopes(fn)
    return contextlib.nullcontext()


def trace_model(fn, *example_args, collapse_modules_after_depth=1, height=800,
                width=None, export_format=None, export_path=None,
                show_constants=False):
    """Trace a JAX model and render its forward pass as an interactive graph.

    Args:
        fn: The model to trace. Pass a Flax NNX module directly to get module
            nesting automatically; for Flax Linen pass ``lambda x: model.apply(params, x)``;
            any callable works and is traced with ``jax.make_jaxpr``.
        *example_args: Example inputs (arrays / pytrees) with the right shapes.
        collapse_modules_after_depth: Nesting depth beyond which modules start collapsed.
        height, width: Rendered graph size in pixels.
        export_format: None (inline display) or one of 'html'/'png'/'svg'.
        export_path: Optional path for HTML export.
        show_constants: Draw inline literal operands and scalar constants as
            Constant nodes (off by default to keep the graph uncluttered).
    """
    if export_format is None and export_path is not None:
        export_format = ExportFormat.HTML
    else:
        export_format = validate_export_format(export_format)

    with _module_scope_context(fn):
        closed_jaxpr = jax.make_jaxpr(fn)(*example_args)
    blobs = build_graph(closed_jaxpr, show_constants=show_constants)

    return plot_graph(
        blobs["adj_list"],
        blobs["module_info"],
        blobs["func_info"],
        blobs["node_to_module_path"],
        blobs["parent_module_to_nodes"],
        blobs["parent_module_to_depth"],
        blobs["graph_node_name_to_without_suffix"],
        blobs["graph_node_display_names"],
        blobs["node_to_attr_name"],
        blobs["ancestor_map"],
        max(collapse_modules_after_depth, 0),
        height,
        width,
        export_format,
        False,
        blobs["repeat_containers"],
        show_modular_view=False,
        export_path=export_path,
    )
