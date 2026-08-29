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
from .hlo_to_graph import build_hlo_graph
from .render import plot_graph, validate_export_format

__all__ = ["trace_model", "ExportFormat"]

# The two program views trace_model can render.
VIEW_GLOBAL = "global"
VIEW_PER_DEVICE = "per_device"
_VIEWS = (VIEW_GLOBAL, VIEW_PER_DEVICE)


def _module_scope_context(fn):
    """Return the explicit module-invocation adapter for this trace."""
    try:
        from flax import nnx
        if isinstance(fn, nnx.Module):
            from .adapters.nnx import named_scopes
            return named_scopes(fn)
    except ImportError:
        pass

    try:
        import equinox as eqx
        if isinstance(fn, eqx.Module):
            from .adapters.equinox import named_scopes
            return named_scopes(fn)
    except ImportError:
        pass

    try:
        from .adapters.linen import named_scopes
        return named_scopes()
    except ImportError:
        pass

    return contextlib.nullcontext()


def _lower_for_hlo(fn, example_args):
    """Lower fn for HLO extraction. For a Flax NNX module the sharded weights must
    be passed as *arguments* (not captured), otherwise they are constant-folded,
    their sharding is lost, and no collectives are inserted."""
    try:
        from flax import nnx
    except ImportError:
        nnx = None

    if nnx is not None and isinstance(fn, nnx.Module):
        from .adapters.nnx import named_scopes

        graphdef, state = nnx.split(fn)

        def forward(state, *args):
            model = nnx.merge(graphdef, state)
            with named_scopes(model):
                return model(*args)

        return jax.jit(forward).lower(state, *example_args)

    with _module_scope_context(fn):
        return jax.jit(fn).lower(*example_args)


def _global_jaxpr_for_hlo(fn, example_args):
    """Trace the unpartitioned program for partitioner-independent shape data."""
    try:
        from flax import nnx
    except ImportError:
        nnx = None

    if nnx is not None and isinstance(fn, nnx.Module):
        graphdef, state = nnx.split(fn)

        def forward(state, *args):
            return nnx.merge(graphdef, state)(*args)

        return jax.make_jaxpr(forward)(state, *example_args)

    return jax.make_jaxpr(fn)(*example_args)


def _active_mesh_shape():
    get_mesh = getattr(jax.sharding, "get_mesh", None)
    if get_mesh is not None:
        try:
            return dict(get_mesh().shape)
        except (AttributeError, RuntimeError):
            pass

    get_abstract_mesh = getattr(jax.sharding, "get_abstract_mesh", None)
    if get_abstract_mesh is not None:
        try:
            shape = dict(get_abstract_mesh().shape)
            return shape or None
        except (AttributeError, RuntimeError):
            pass

    return None


def trace_model(fn, *example_args, view=VIEW_GLOBAL, collapse_modules_after_depth=1,
                height=800, width=None, export_format=None, export_path=None,
                show_constants=False, return_html=False):
    """Trace a JAX model and render its forward pass as an interactive graph.

    Args:
        fn: The model to trace. Pass a Flax NNX module directly to get module
            nesting automatically; for Flax Linen pass ``lambda x: model.apply(params, x)``;
            any callable works and is traced with ``jax.make_jaxpr``.
        *example_args: Example inputs (arrays / pytrees) with the right shapes.
        view: Which program view to render. ``"global"`` (default) shows the
            jaxpr as written, using logical/global tensor shapes before partitioning.
            ``"per_device"`` shows the post-partitioning HLO executed by each
            device, including local tensor shapes and compiler-inserted collectives
            (all-reduce/all-gather/...). Trace under a mesh so there is sharding for
            the compiler to act on.
        collapse_modules_after_depth: Nesting depth beyond which modules start collapsed.
        height, width: Rendered graph size in pixels.
        export_format: None (inline display) or one of 'html'/'png'/'svg'.
        export_path: Optional path for HTML export.
        show_constants: Draw inline literal operands and scalar constants as
            Constant nodes (off by default to keep the graph uncluttered). Global view only.
        return_html: Return the embeddable graph HTML instead of displaying or exporting it.
    """
    if view not in _VIEWS:
        raise ValueError(f"Invalid view: {view!r}. Must be one of {_VIEWS}.")

    if export_format is None and export_path is not None:
        export_format = ExportFormat.HTML
    else:
        export_format = validate_export_format(export_format)

    if view == VIEW_PER_DEVICE:
        global_jaxpr = _global_jaxpr_for_hlo(fn, example_args)
        lowered = _lower_for_hlo(fn, example_args)
        blobs = build_hlo_graph(
            lowered,
            mesh_shape=_active_mesh_shape(),
            global_jaxpr=global_jaxpr,
        )
    else:
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
        return_html=return_html,
    )
