"""Convert a JAX function's jaxpr into the graph data used by the frontend.

A jaxpr is already a dataflow DAG: each equation is a primitive with SSA-named
input/output variables, so edges follow directly from variable dependencies, and
every variable carries a static shape and dtype (its ``aval``).

Nesting comes from each equation's ``name_stack`` (e.g. ``MLP/Dense_0``). Frameworks
that use named scopes (Flax Linen, Haiku) populate it; those that do not (raw JAX,
Equinox, NNX) produce a flat graph.

Only leaf nodes are emitted into ``adj_list``; the frontend synthesizes the
collapsible module containers from ``ancestor_map``.
"""
import re
from collections import defaultdict

import jax
from jax import core as jax_core

from .enums import NodeType
from .graph_transforms import build_immediate_ancestor_map


def _shape_str(aval):
    shape = getattr(aval, "shape", None)
    if shape is None:
        return "( )"
    if len(shape) == 0:
        return "( )"
    return f"({', '.join(str(d) for d in shape)})"


def _scope_parts(eqn):
    """Return the name_stack of an equation as a list of scope names, root last.

    e.g. name_stack 'MLP/Dense_0' -> ['Dense_0', 'MLP']  (immediate parent first).
    Empty name_stack -> [] (node lives at top level, no nesting).
    """
    ns = str(getattr(eqn.source_info, "name_stack", "") or "")
    parts = [p for p in ns.split("/") if p]
    return parts


def _safe_id(s):
    """Make a string usable as a Graphviz/DOT node id (frontend renders via DOT).

    DOT rejects '/' , '.' etc. in unquoted ids, so container ids built from scope
    paths like 'MLP/Dense_0' or 'eqx.nn.Linear' must be sanitized to [0-9A-Za-z_].
    """
    return re.sub(r"[^0-9A-Za-z_]", "_", s)


def _scope_containers(parts):
    """Turn ['MLP', 'Dense_0'] (root first) into [(container_id, label), ...],
    immediate-first.

    container_id is DOT-safe and unique to the cumulative path (so distinct scopes
    never collide); label is the human name of that scope level.
      ['MLP', 'Dense_0'] -> [('scope_MLP_Dense_0', 'Dense_0'), ('scope_MLP', 'MLP')]
    """
    levels = []
    for i in range(1, len(parts) + 1):
        cid = "scope_" + _safe_id("/".join(parts[:i]))
        levels.append((cid, parts[i - 1]))
    return levels[::-1]  # immediate parent first, root last


def _pretty_primitive_name(eqn):
    """Human-friendlier name for a primitive equation.

    Higher-order primitives (pjit/custom_jvp_call) often wrap a named function
    (e.g. relu). Surface that name when cheaply available, else the primitive name.
    """
    params = eqn.params
    name = params.get("name")
    if isinstance(name, str) and name:
        return name
    if eqn.primitive.name == "custom_jvp_call":
        call_jaxpr = params.get("call_jaxpr")
        inner = getattr(call_jaxpr, "jaxpr", call_jaxpr)
        if inner is not None:
            for sub in inner.eqns:
                inner_name = sub.params.get("name")
                if isinstance(inner_name, str) and inner_name:
                    return inner_name
    return eqn.primitive.name


def build_graph(closed_jaxpr):
    """Walk a ClosedJaxpr and produce the torchvista frontend blobs.

    Returns a dict of all structures plot_graph expects.
    """
    jaxpr = closed_jaxpr.jaxpr

    adj_list = {}
    func_info = {}
    module_info = {}
    graph_node_display_names = {}
    graph_node_name_to_without_suffix = {}
    node_to_module_path = {}
    node_to_ancestors = {}
    parent_module_to_nodes = defaultdict(list)
    parent_module_to_depth = {}

    # Map each jaxpr Var -> the node name that produces it.
    var_to_source = {}
    counter = [0]

    def new_node(base):
        counter[0] += 1
        # Sanitize the id (DOT-safe) but keep the raw base for the display name.
        return f"{_safe_id(base)}_{counter[0]}"

    def register_container_labels(containers):
        # containers is [(container_id, label), ...] immediate-first
        for cid, label in containers:
            if cid in graph_node_display_names:
                continue
            graph_node_display_names[cid] = label
            # Strip a trailing _<idx> so the module *type* reads cleanly (Dense_0 -> Dense)
            without = label
            if "_" in label and label.rsplit("_", 1)[-1].isdigit():
                without = label.rsplit("_", 1)[0]
            graph_node_name_to_without_suffix[cid] = without
            module_info[cid] = {"type": without, "parameters": {}, "attributes": {}}

    seen_edges = set()

    def add_edge(src_node, dst_node, var):
        # edge_data_id identifies the *tensor* (the jaxpr Var, shared across all its
        # uses), so the frontend can merge edges carrying the same tensor into a
        # collapsed container while keeping distinct tensors separate.
        edge_data_id = id(var)
        edge_key = (src_node, dst_node, edge_data_id)
        if edge_key in seen_edges:
            return
        seen_edges.add(edge_key)
        adj_list[src_node]["edges"].append({
            "target": dst_node,
            "dims": _shape_str(var.aval),
            "edge_data_id": edge_data_id,
        })

    # --- Inputs: the jaxpr's invars are the real function arguments ---
    for i, var in enumerate(jaxpr.invars):
        node = f"input_{i}"
        adj_list[node] = {"edges": [], "failed": False, "node_type": NodeType.INPUT.value}
        graph_node_display_names[node] = f"input_{i}"
        graph_node_name_to_without_suffix[node] = "input"
        node_to_ancestors[node] = []
        var_to_source[var] = node

    # --- Params/consts: closed-over arrays (e.g. model weights) ---
    for i, var in enumerate(jaxpr.constvars):
        node = f"param_{i}"
        adj_list[node] = {"edges": [], "failed": False, "node_type": NodeType.PARAMETER.value}
        graph_node_display_names[node] = "param"
        graph_node_name_to_without_suffix[node] = "param"
        # Ancestors are assigned lazily to the scope of the first consuming eqn,
        # so weights nest inside the module that uses them.
        node_to_ancestors[node] = []
        var_to_source[var] = node

    def maybe_assign_param_scope(src_node, container_ids):
        if src_node.startswith("param_") and not node_to_ancestors.get(src_node):
            node_to_ancestors[src_node] = list(container_ids)

    # --- Equations become leaf nodes ---
    for eqn in jaxpr.eqns:
        parts = _scope_parts(eqn)          # root first
        containers = _scope_containers(parts)          # [(id, label), ...] immediate first
        container_ids = [cid for cid, _ in containers]
        register_container_labels(containers)

        base = _pretty_primitive_name(eqn)
        node = new_node(base)
        adj_list[node] = {"edges": [], "failed": False, "node_type": NodeType.OPERATION.value}
        graph_node_display_names[node] = base
        graph_node_name_to_without_suffix[node] = base
        node_to_module_path[node] = eqn.primitive.name
        node_to_ancestors[node] = list(container_ids)

        # Record this node under every ancestor container, tracking each container's
        # subtree depth (distance down to its deepest leaf).
        depth = 1
        for cid in container_ids:
            parent_module_to_nodes[cid].append(node)
            parent_module_to_depth[cid] = max(depth, parent_module_to_depth.get(cid, 0))
            depth += 1

        # Edges in: connect every non-literal input var to this node
        for invar in eqn.invars:
            if isinstance(invar, jax_core.Literal):
                continue
            src = var_to_source.get(invar)
            if src is None:
                continue
            maybe_assign_param_scope(src, container_ids)
            add_edge(src, node, invar)

        # Params of the op (dimension_numbers etc.) shown on click
        func_info[node] = {
            "positional_args": [],
            "keyword_args": {k: str(v)[:80] for k, v in eqn.params.items()
                             if k not in ("jaxpr", "call_jaxpr", "branches", "jvp_jaxpr_thunk")},
        }

        # Outputs: this node produces these vars
        for outvar in eqn.outvars:
            if isinstance(outvar, jax_core.DropVar):
                continue
            var_to_source[outvar] = node

    # --- Outputs ---
    for i, outvar in enumerate(jaxpr.outvars):
        node = f"output_{i}"
        adj_list[node] = {"edges": [], "failed": False, "node_type": NodeType.OUTPUT.value}
        graph_node_display_names[node] = f"output_{i}"
        graph_node_name_to_without_suffix[node] = "output"
        node_to_ancestors[node] = []
        if isinstance(outvar, jax_core.Literal):
            continue
        src = var_to_source.get(outvar)
        if src is not None:
            add_edge(src, node, outvar)

    ancestor_map = build_immediate_ancestor_map(node_to_ancestors, adj_list)

    return {
        "adj_list": adj_list,
        "module_info": module_info,
        "func_info": func_info,
        "node_to_module_path": node_to_module_path,
        "parent_module_to_nodes": dict(parent_module_to_nodes),
        "parent_module_to_depth": parent_module_to_depth,
        "graph_node_name_to_without_suffix": graph_node_name_to_without_suffix,
        "graph_node_display_names": graph_node_display_names,
        "node_to_attr_name": {},
        "ancestor_map": ancestor_map,
        "repeat_containers": [],
    }
