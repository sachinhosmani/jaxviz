"""Read the post-partitioning, pre-fusion HLO and build the frontend blobs.

Under Auto/school-1 sharding the collectives don't exist in the jaxpr — XLA inserts
them while partitioning. We dump HLO after the SPMD partitioner (before fusion) and
select the module by content (earliest dump containing a collective).

Attribution uses ONLY first-party metadata:
  * op_name scopes (e.g. jit(forward)/jit(relu)/max  ->  label "relu")
  * the HLO stack-frame source-location index (file/function/line/column),
    which maps each op to the exact call site in the user's model code
  * for weights, the pytree path in op_name (state['dense0']['kernel'].value)

Module nesting is derived from the stack-frame chain: each `X.__call__` frame is a
module boundary, and the attribute name (dense0/dense1) is read from the *source
line* at the call-site column recorded in the frame table. If a node has no
reliable frame/scope, it is left top-level (unattributed) rather than guessed.
"""
import ast
import glob
import os
import re
import tempfile
from collections import defaultdict

from .enums import NodeType

COLLECTIVE_OPCODES = {
    "all-reduce", "all-gather", "all-to-all", "reduce-scatter",
    "collective-permute", "collective-broadcast", "ragged-all-to-all",
    "all-reduce-start", "all-gather-start", "collective-permute-start",
}

_SIG_PARAM_RE = re.compile(r"([\w.\-]+):\s*([a-z0-9]+\[[^\]]*\](?:\{[^}]*\})?|\(\))")
_INSTR_RE = re.compile(r"^\s*(?:ROOT\s+)?%([\w.\-]+)\s*=\s*(\S+)\s+([\w\-]+)\((.*)$")
_OPERAND_RE = re.compile(r"%[\w.\-]+")
_OP_NAME_RE = re.compile(r'op_name="((?:[^"\\]|\\.)*)"')
_STACK_ID_RE = re.compile(r"stack_frame_id=(\d+)")
_CHANNEL_RE = re.compile(r"channel_id=(\d+)")
_RGROUPS_RE = re.compile(r"replica_groups=((?:mesh\[[^\]]*\]\s*\{[^}]*\})|\{\{[^}]*\}[^,]*\})")
_SHARDING_RE = re.compile(r"sharding=(\{[^}]*\})")
_SPMD_RE = re.compile(r'is_spmd_generated="true"')

# state['dense0']['kernel'].value  and  state["dense0"]["kernel"].value
_STATE_PATH_RE = re.compile(r"""(?:state|params)((?:\[\s*['"][^'"]+['"]\s*\])+)""")
_BRACKET_KEY_RE = re.compile(r"""\[\s*['"]([^'"]+)['"]\s*\]""")


def _safe_id(s):
    return re.sub(r"[^0-9A-Za-z_]", "_", s)


def _hlo_shape_to_dims(shape):
    m = re.search(r"\[([^\]]*)\]", shape or "")
    if not m or not m.group(1).strip():
        return "( )"
    return "(" + ", ".join(p.strip() for p in m.group(1).split(",")) + ")"


def _hlo_shape_parts(shape):
    """'f32[16,8]{1,0}' -> ('f32', [16, 8]); scalar 'f32[]' -> ('f32', [])."""
    dtype_m = re.match(r"\s*([a-z0-9]+)", shape or "")
    dtype = dtype_m.group(1) if dtype_m else ""
    dims_m = re.search(r"\[([^\]]*)\]", shape or "")
    dims = ([int(x) for x in dims_m.group(1).split(",")]
            if dims_m and dims_m.group(1).strip() else [])
    return dtype, dims


def _partition_axes(partitions, mesh_shape):
    """Map each tensor partition factor to named mesh axes when unambiguous."""
    if not mesh_shape:
        return None

    mesh_axes = [
        (str(name), int(size)) for name, size in mesh_shape.items()
        if int(size) > 1
    ]
    solutions = []

    def visit(dimension, remaining_axes, assignment):
        if len(solutions) > 1:
            return
        if dimension == len(partitions):
            solutions.append([
                [{"name": name, "size": size} for name, size in axes]
                for axes in assignment
            ])
            return

        target = partitions[dimension]
        if target == 1:
            visit(dimension + 1, remaining_axes, assignment + [()])
            return

        axis_count = len(remaining_axes)
        for mask in range(1, 1 << axis_count):
            selected = [remaining_axes[index] for index in range(axis_count) if mask & (1 << index)]
            product = 1
            for _, size in selected:
                product *= size
            if product != target:
                continue
            selected_indexes = {index for index in range(axis_count) if mask & (1 << index)}
            unused = [axis for index, axis in enumerate(remaining_axes) if index not in selected_indexes]
            visit(
                dimension + 1,
                unused,
                assignment + [tuple(selected)],
            )

    visit(0, mesh_axes, [])
    return solutions[0] if len(solutions) == 1 else None


def _shape_info(local_dims, global_dims, mesh_shape=None):
    """Structured shape data used by the per-device edge visualization."""
    local = list(local_dims)
    if global_dims is None:
        return {
            "global": None,
            "local": local,
            "partitions": None,
            "axes": None,
            "status": "unavailable",
        }

    global_shape = list(global_dims)
    if len(global_shape) != len(local):
        return {
            "global": None,
            "local": local,
            "partitions": None,
            "axes": None,
            "status": "unavailable",
        }

    partitions = []
    for global_dim, local_dim in zip(global_shape, local):
        if global_dim == local_dim:
            partitions.append(1)
        elif local_dim > 0 and global_dim % local_dim == 0:
            partitions.append(global_dim // local_dim)
        else:
            return {
                "global": None,
                "local": local,
                "partitions": None,
                "axes": None,
                "status": "unavailable",
            }

    return {
        "global": global_shape,
        "local": local,
        "partitions": partitions,
        "axes": _partition_axes(partitions, mesh_shape),
        "status": "verified",
    }


def _sharding_tiling(sharding):
    """Per-dimension split factors from a sharding annotation. '{replicated}' (or
    none) -> None, meaning global == local. Returns None if it can't be parsed."""
    if not sharding or "replicated" in sharding:
        return None
    m = re.search(r"devices=\[([\d,]+)\]", sharding)
    if not m:
        return None
    tiling = [int(x) for x in m.group(1).split(",")]
    if "last_tile_dim_replicate" in sharding:
        tiling = tiling[:-1]   # trailing group is replication, not a data dim
    return tiling


def _global_from_sharding(local_dims, sharding):
    """Global (unsharded) shape = local shape x the sharding's per-dim tiling."""
    tiling = _sharding_tiling(sharding)
    if tiling is None:
        return list(local_dims)
    if len(tiling) != len(local_dims):
        return None
    return [ld * t for ld, t in zip(local_dims, tiling)]


def _local_from_global(global_dims, sharding):
    """Local (per-device) shape = global shape / the sharding's per-dim tiling.
    Used to self-check a global shape against the local shape actually rendered:
    if they don't reconcile, the global is not trusted. Returns None on mismatch."""
    tiling = _sharding_tiling(sharding)
    if tiling is None:
        return list(global_dims)
    if len(tiling) != len(global_dims):
        return None
    out = []
    for g, t in zip(global_dims, tiling):
        if t == 0 or g % t:
            return None
        out.append(g // t)
    return out


def _parse_scalar_const(literal, dtype):
    """Turn an HLO scalar constant literal into a plain Python value."""
    literal = literal.strip()
    try:
        if dtype == "pred":
            return literal == "true"
        if dtype[:1] in ("s", "u"):
            return int(literal)
        return float(literal)
    except (ValueError, IndexError):
        return literal


def _operand_arg(hlo_name, name_to_node, out_shape, const_values):
    """Describe one operand the way torchvista formats args: a scalar constant is
    shown as its raw value; anything else as a tensor {shape, dtype}."""
    node = name_to_node.get(hlo_name)
    if node is not None and node in const_values:
        return const_values[node]                 # scalar literal -> raw value
    dtype, dims = _hlo_shape_parts(out_shape.get(node, "")) if node else ("", [])
    return {"_type": "tensor", "shape": dims, "dtype": dtype}


def _split_top_commas(s):
    """Split on commas that are not inside (), [] or {}."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


# Not call parameters — placement/metadata, excluded from keyword_args.
_ATTR_EXCLUDE = {"metadata", "sharding", "backend_config", "frontend_attributes"}


def _parse_hlo_attrs(tail):
    """Parse an op's genuine static parameters (dimensions, contracting dims,
    replica_groups, channel_id, ...) from the HLO attribute tail after the
    operand list. Placement/metadata (op_name, source, sharding) are excluded —
    those are not arguments the op was called with."""
    attrs = {}
    for piece in _split_top_commas(tail):
        piece = piece.strip().strip(",").strip()
        if "=" not in piece:
            continue
        key, val = piece.split("=", 1)
        key = key.strip()
        if key in _ATTR_EXCLUDE or not re.fullmatch(r"[a-z_][a-z0-9_]*", key):
            continue
        attrs[key] = val.strip()
    return attrs


def _unescape(s):
    # HLO escapes single quotes in op_name as \'
    return s.replace("\\'", "'").replace('\\"', '"') if s else s


# --------------------------------------------------------------------------
# Dump + module selection (unchanged strategy: content-based)
# --------------------------------------------------------------------------
def _dump_hlo_stages(lowered):
    """Compile with dumping and return two stages:
      * rendered   -- post-partition module: per-device (local) shapes + the
                      inserted collectives (this is the graph we draw);
      * propagated -- post sharding-propagation, pre-partition module: every op's
                      GLOBAL shape + its sharding (source of the global shapes),
                      or None if that stage isn't available.
    """
    dump_dir = tempfile.mkdtemp(prefix="jaxviz_hlo_")
    lowered.compile(compiler_options={
        "xla_dump_to": dump_dir,
        "xla_dump_hlo_pass_re": "spmd.*|shardy.*",
        "xla_dump_hlo_as_text": True,
    })
    coll = re.compile(r"\b(" + "|".join(re.escape(o) for o in COLLECTIVE_OPCODES) + r")\(")

    def is_module(p):
        b = os.path.basename(p)
        return not any(x in b for x in ("buffer-assignment", "live-range", "memory-usage"))

    candidates = [p for p in sorted(glob.glob(os.path.join(dump_dir, "*.txt"))) if is_module(p)]
    with_coll = [p for p in candidates if coll.search(open(p).read())]
    if with_coll:
        rendered = with_coll[0]
    elif candidates:
        rendered = candidates[-1]
    else:
        raise RuntimeError(f"No HLO dump produced in {dump_dir}")

    # last module before partitioning == fully sharding-propagated, still global
    propagated = None
    for p in candidates:
        if "before_spmd-partitioning" in os.path.basename(p):
            propagated = p
    return open(rendered).read(), (open(propagated).read() if propagated else None)


def _build_global_index(propagated_text):
    """From the post-propagation stage, map (line, col, opcode) -> list of
    (global_shape, sharding). Keyed by source location + opcode so it can be
    joined to the rendered graph and self-checked against the local shape."""
    if not propagated_text:
        return {}
    sfi = _StackFrameIndex(propagated_text)
    ufiles = {f for f in sfi.files.values()
              if not f.startswith("<") and "site-packages" not in f
              and "dist-packages" not in f}
    _, body = _entry_block(propagated_text)
    index = {}
    for ln in body:
        m = _INSTR_RE.match(ln)
        if not m or m.group(3) == "parameter":
            continue
        _, opcode, shape, rest = m.group(1), m.group(3), m.group(2), m.group(4)
        _, gdims = _hlo_shape_parts(shape)
        shm = _SHARDING_RE.search(ln)
        sharding = shm.group(1) if shm else None
        loc = None
        if sid := _STACK_ID_RE.search(ln):
            for f, fn, l, c in sfi.resolve(int(sid.group(1))):
                if f in ufiles:
                    loc = (l, c)
                    break
        if loc is not None:
            index.setdefault((loc[0], loc[1], opcode), []).append((tuple(gdims), sharding))
    return index


def _propagated_output(propagated_text):
    """Global shape + sharding of the model's output, from the post-propagation
    stage's ROOT (its output IS the function's recorded result). Returns
    (global_shape, sharding) for a single-array output, else None."""
    if not propagated_text:
        return None
    _, body = _entry_block(propagated_text)
    for ln in body:
        if not ln.lstrip().startswith("ROOT"):
            continue
        m = _INSTR_RE.match(ln)
        if not m:
            return None
        shape = m.group(2)
        if shape.strip().startswith("("):   # tuple output -> not handled here
            return None
        _, gdims = _hlo_shape_parts(shape)
        shm = _SHARDING_RE.search(ln)
        return (tuple(gdims), shm.group(1) if shm else None)
    return None


# --------------------------------------------------------------------------
# Stack-frame index: parse the header tables and resolve frame -> [frames]
# --------------------------------------------------------------------------
class _StackFrameIndex:
    """Resolves stack_frame_id -> list of Frame(file, function, line, column),
    innermost first. Handles the text printer's +1 parent-id shift, verified
    against the proto encoding of the same module."""

    def __init__(self, text):
        self.files = self._string_table(text, "FileNames")
        self.funcs = self._string_table(text, "FunctionNames")
        self.locs = {}
        for k, body in self._brace_table(text, "FileLocations").items():
            fn = int(re.search(r"file_name_id=(\d+)", body).group(1))
            fu = int(re.search(r"function_name_id=(\d+)", body).group(1))
            line = int(re.search(r"line=(\d+)", body).group(1))
            col_m = re.search(r"(?<!end_)column=(\d+)", body)
            col = int(col_m.group(1)) if col_m else 0
            self.locs[k] = (fn, fu, line, col)
        self.frames = {}
        shifted = False
        for k, body in self._brace_table(text, "StackFrames").items():
            lid = int(re.search(r"file_location_id=(\d+)", body).group(1))
            pid_m = re.search(r"parent_frame_id=(\d+)", body)
            pid = int(pid_m.group(1)) if pid_m else 0
            self.frames[k] = (lid, pid)
            if k == pid:
                shifted = True
        self._shift = 1 if shifted else 0

    @staticmethod
    def _string_table(text, name):
        rows, grab = {}, False
        for ln in text.splitlines():
            if ln.strip() == name:
                grab = True
                continue
            if grab:
                m = re.match(r'^(\d+)\s+"(.*)"\s*$', ln.strip())
                if not m:
                    break
                rows[int(m.group(1))] = m.group(2)
        return rows

    @staticmethod
    def _brace_table(text, name):
        rows, grab = {}, False
        for ln in text.splitlines():
            if ln.strip() == name:
                grab = True
                continue
            if grab:
                m = re.match(r"^(\d+)\s+(\{.*\})\s*$", ln.strip())
                if not m:
                    break
                rows[int(m.group(1))] = m.group(2)
        return rows

    def resolve(self, frame_id):
        out, seen = [], set()
        while frame_id and frame_id in self.frames and frame_id not in seen:
            seen.add(frame_id)
            lid, pid = self.frames[frame_id]
            fn, fu, line, col = self.locs[lid]
            out.append((self.files.get(fn, "?"), self.funcs.get(fu, "?"), line, col))
            frame_id = pid - self._shift
        return out


# --------------------------------------------------------------------------
# Source reader: recover the attribute name at a call site (dense0/dense1)
# Location comes from XLA's frame table; source is the user's own file.
# Parsed with `ast`, no regex guessing on the source.
# --------------------------------------------------------------------------
class _SourceResolver:
    def __init__(self):
        self._lines_cache = {}

    def _lines(self, path):
        if path not in self._lines_cache:
            try:
                with open(path) as f:
                    self._lines_cache[path] = f.read().splitlines()
            except OSError:
                self._lines_cache[path] = None
        return self._lines_cache[path]

    def attr_at(self, file, line, column):
        """Return the attribute name of the call whose expression starts at
        (line, column): for `self.dense0(x)` -> 'dense0'; for `nnx.relu(x)`
        -> None (not a self attribute -> not a submodule instance).
        Returns None on any uncertainty."""
        lines = self._lines(file)
        if not lines or not (1 <= line <= len(lines)):
            return None
        src = lines[line - 1]
        # find the smallest Call node whose func starts at the given column
        try:
            tree = ast.parse(src.strip())
        except SyntaxError:
            return None
        target_col = column - (len(src) - len(src.lstrip()))
        best = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fcol = getattr(node.func, "col_offset", None)
                if fcol is None:
                    continue
                if abs(fcol - target_col) <= 1:
                    best = node.func
                    break
        if best is None:
            return None
        # self.dense0(...)  -> Attribute(value=Name('self'), attr='dense0')
        if isinstance(best, ast.Attribute) and isinstance(best.value, ast.Name) \
                and best.value.id == "self":
            return best.attr
        return None


# --------------------------------------------------------------------------
# op_name -> clean op label   (jit(forward)/jit(relu)/max -> relu)
# --------------------------------------------------------------------------
def _opname_scopes(op_name, wrapper=None):
    """The jit(<name>) scope segments in op_name (e.g. jit(forward)/jit(relu)/max
    -> ['relu']), minus the outer wrapper (the lowered fn, here 'forward').

    These scopes are JAX's own recorded function names and act as *containers*:
    a compound op like relu lowers to several primitives (broadcast, maximum) that
    all share the jit(relu) scope, so we nest them under a 'relu' box and keep each
    primitive's real opcode as its label — rather than mislabeling them 'relu'.
    Bare primitives (dot, add) have no jit(<name>) scope and stay un-nested."""
    if not op_name:
        return []
    names = re.findall(r"jit\(([^)]+)\)", op_name)   # e.g. ["forward", "relu"]
    if wrapper is not None:
        names = [n for n in names if n != wrapper]
    return names


# --------------------------------------------------------------------------
# Module path from the frame chain.
# Each `Class.__call__` frame is a module boundary; the submodule *name* is
# read from the source at the call-site column of the frame ONE LEVEL DOWN
# the stack (the caller that invoked that __call__).
# --------------------------------------------------------------------------
def _module_path_from_frames(frames, src, user_files):
    """frames: innermost-first list of (file, func, line, col).
    Returns module names (outermost first), or [] if none resolvable.

    A submodule instance is invoked as `self.<name>(...)` in user code, and each
    such call is a frame in the op's stack. So we read the attribute name at each
    *user* frame's recorded (line, column): a frame that sits on `self.dense0(x)`
    yields 'dense0'; a frame on a plain function call (`nnx.relu(x)`) yields None
    and is skipped (relu is not a submodule instance, so it stays un-nested).
    Nested submodules (`self.block(x)` -> `self.dense0(x)`) produce a multi-level
    path. Anything unreadable is skipped rather than guessed.
    """
    path = []
    for (f, fn, ln, col) in frames:   # innermost first
        if f not in user_files:
            continue
        attr = src.attr_at(f, ln, col)
        if attr:
            path.append(attr)
    return list(reversed(path))       # outermost first


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_hlo_graph(lowered, mesh_shape=None):
    text, propagated = _dump_hlo_stages(lowered)
    global_index = _build_global_index(propagated)
    out_global = _propagated_output(propagated)
    sfi = _StackFrameIndex(text)
    src = _SourceResolver()
    user_files = {f for f in sfi.files.values()
                  if not f.startswith("<") and "site-packages" not in f
                  and "dist-packages" not in f}

    sig_params, body = _entry_block(text)

    adj_list, func_info = {}, {}
    graph_node_display_names, graph_node_name_to_without_suffix = {}, {}
    node_to_module_path = {}
    node_to_attr_name = {}
    name_to_node, out_shape, pending_edges = {}, {}, []
    const_values = {}  # node_id -> raw scalar value (for scalar constants)
    node_global = {}   # node_id -> global (logical) output shape, when known
    node_modpath = {}  # node_id -> list[str] (outermost first) or []

    def add_node(node_id, node_type, label, without=None):
        adj_list[node_id] = {"edges": [], "failed": False, "node_type": node_type}
        graph_node_display_names[node_id] = label
        graph_node_name_to_without_suffix[node_id] = without or label

    # The outer wrapper is the first jit(<name>) segment shared by ops
    # (here 'forward', from the lowered fn). Detect it instead of hardcoding.
    wrapper = None
    for ln in body:
        on = _OP_NAME_RE.search(ln)
        if on:
            outer = re.match(r"jit\(([^)]+)\)", on.group(1))
            if outer:
                wrapper = outer.group(1)
                break

    # The root container is the model's top module: the outermost user-code
    # `<Class>.__call__` frame (e.g. MLP), so the whole graph nests under it like
    # the global graph does. Detected once from any op's frame chain.
    root_class = None
    for ln in body:
        sid_m = _STACK_ID_RE.search(ln)
        if not sid_m:
            continue
        frames = sfi.resolve(int(sid_m.group(1)))
        for (f, fn, l, c) in reversed(frames):   # outermost first
            if f in user_files and fn.endswith(".__call__"):
                root_class = fn[:-len(".__call__")]
                break
        if root_class:
            break
    root_prefix = [root_class] if root_class else []

    # ---- pass 1: parameters (op_name carries input name or state[...] path) ----
    param_meta = {}
    for ln in body:
        m = _INSTR_RE.match(ln)
        if m and m.group(3) == "parameter":
            on = _OP_NAME_RE.search(ln)
            sh = _SHARDING_RE.search(ln)
            param_meta[m.group(1)] = (_unescape(on.group(1)) if on else None,
                                      sh.group(1) if sh else None)

    for pname, pshape in sig_params:
        node_id = _safe_id(pname)
        opname, sharding = param_meta.get(pname, (None, None))
        state_path = _state_path(opname)
        if state_path is not None:
            # a weight: type Parameter, label from the pytree path, nest by it
            leaf = ".".join(state_path)  # dense0.kernel
            add_node(node_id, NodeType.PARAMETER.value, leaf, "param")
            node_modpath[node_id] = root_prefix + state_path[:-1]  # e.g. [MLP, dense0]
            node_to_attr_name[node_id] = state_path[-1]
        elif opname and "/" not in opname and not opname.startswith("jit("):
            # a genuine forward input (args[0])
            add_node(node_id, NodeType.INPUT.value, opname, "input")
            node_modpath[node_id] = []
        else:
            add_node(node_id, NodeType.PARAMETER.value, "param", "param")
            node_modpath[node_id] = []
        # leaves: nothing was "called with" them, and no static params
        func_info[node_id] = {"positional_args": [], "keyword_args": {}}
        name_to_node[pname] = node_id
        out_shape[node_id] = pshape
        # global (unsharded) shape from local x the sharding tiling
        _, local_dims = _hlo_shape_parts(pshape)
        g = _global_from_sharding(local_dims, sharding)
        if g is not None:
            node_global[node_id] = g

    # ---- pass 2: instructions ----
    root_name = None
    for ln in body:
        m = _INSTR_RE.match(ln)
        if not m:
            continue
        name, shape, opcode, rest = m.groups()
        if opcode == "parameter":
            continue
        node_id = _safe_id(name)
        is_coll = opcode in COLLECTIVE_OPCODES
        on = _OP_NAME_RE.search(ln)
        op_name = _unescape(on.group(1)) if on else None

        operand_str, _, attr_tail = rest.partition(")")

        # Label every op by its real (low-level) opcode. Friendly names like relu
        # are represented as *containers* (via op_name scopes), not labels.
        if opcode == "constant":
            add_node(node_id, NodeType.CONSTANT.value, "constant", "constant")
            dtype, dims = _hlo_shape_parts(shape)
            if not dims:  # a scalar constant -> remember its value (shown raw)
                const_values[node_id] = _parse_scalar_const(operand_str, dtype)
        else:
            add_node(node_id, NodeType.OPERATION.value, opcode, opcode)

        name_to_node[name] = node_id
        out_shape[node_id] = shape
        operands = [op.lstrip("%") for op in _OPERAND_RE.findall(operand_str)
                    if op != "%" + name]
        for op in operands:
            pending_edges.append((op, node_id))

        # positional_args = what this node was actually called with (its operands);
        # keyword_args = the op's genuine static parameters (no metadata).
        func_info[node_id] = {
            "positional_args": [_operand_arg(o, name_to_node, out_shape, const_values)
                                for o in operands],
            "keyword_args": _parse_hlo_attrs(attr_tail),
        }

        # Container path = root module (MLP) + submodule from stack frames
        # (dense0/dense1, via self.<attr>) + op_name scope (relu). All from
        # first-party metadata; anything unresolved is simply omitted.
        sub_path = []
        node_loc = None
        if sid_m := _STACK_ID_RE.search(ln):
            frames = sfi.resolve(int(sid_m.group(1)))
            sub_path = _module_path_from_frames(frames, src, user_files)
            for f, fn, l, c in frames:
                if f in user_files:
                    node_loc = (l, c)
                    break

        # Global shape from the post-propagation stage, joined by (line, col, opcode)
        # and SELF-CHECKED: keep it only if global / sharding == the local shape we
        # render. Collectives have no counterpart there, so they get nothing (right).
        if node_loc is not None:
            _, local_dims = _hlo_shape_parts(shape)
            verified = set()
            for gdims, gsharding in global_index.get((node_loc[0], node_loc[1], opcode), []):
                expected = _local_from_global(gdims, gsharding)
                if expected is not None and list(expected) == list(local_dims):
                    verified.add(tuple(gdims))
            if len(verified) == 1:
                node_global[node_id] = list(next(iter(verified)))

        scope_path = _opname_scopes(op_name, wrapper)
        node_modpath[node_id] = root_prefix + sub_path + scope_path

        if ln.lstrip().startswith("ROOT"):
            root_name = name

    def _edge(src_node, dst_node):
        # Keep shape data structured so the frontend can draw the global-to-local
        # partitioning instead of flattening it into a long text label.
        local_str = _hlo_shape_to_dims(out_shape.get(src_node, ""))
        _, local_dims = _hlo_shape_parts(out_shape.get(src_node, ""))
        g = node_global.get(src_node)
        return {
            "target": dst_node,
            "dims": local_str,
            "shape_info": _shape_info(local_dims, g, mesh_shape),
            "edge_data_id": f"{src_node}->{dst_node}",
        }

    # ---- edges ----
    seen = set()
    for src_name, dst_node in pending_edges:
        src_node = name_to_node.get(src_name)
        if src_node is None or src_node not in adj_list:
            continue
        if (src_node, dst_node) in seen:
            continue
        seen.add((src_node, dst_node))
        adj_list[src_node]["edges"].append(_edge(src_node, dst_node))

    if root_name is not None:
        add_node("output_0", NodeType.OUTPUT.value, "output_0", "output")
        node_modpath["output_0"] = []
        rn = name_to_node[root_name]
        # The output tensor's global shape comes from the recorded output interface
        # (post-propagation ROOT), self-checked against the rendered local shape. This
        # is why a collective producing the output still yields a global on its output
        # edge, even though collectives otherwise get none.
        if out_global is not None and rn not in node_global:
            gdims, gsharding = out_global
            expected = _local_from_global(gdims, gsharding)
            _, root_local = _hlo_shape_parts(out_shape.get(rn, ""))
            if expected is not None and list(expected) == list(root_local):
                node_global[rn] = list(gdims)
        adj_list[rn]["edges"].append(_edge(rn, "output_0"))

    # ---- build the module containers from node_modpath ----
    (ancestor_map, parent_module_to_nodes, parent_module_to_depth,
     module_info, node_to_module_path) = _build_hierarchy(node_modpath, adj_list)

    # A container's display label is its module name (dense1), not its raw id
    # (mod_MLP_dense1) — mirror what the global graph registers.
    for mid, info in module_info.items():
        graph_node_display_names[mid] = info["name"]
        graph_node_name_to_without_suffix[mid] = info["name"]

    return {
        "adj_list": adj_list, "module_info": module_info, "func_info": func_info,
        "node_to_module_path": node_to_module_path,
        "parent_module_to_nodes": parent_module_to_nodes,
        "parent_module_to_depth": parent_module_to_depth,
        "graph_node_name_to_without_suffix": graph_node_name_to_without_suffix,
        "graph_node_display_names": graph_node_display_names,
        "node_to_attr_name": node_to_attr_name, "ancestor_map": ancestor_map,
        "repeat_containers": [],
    }


def _state_path(op_name):
    """state['dense0']['kernel'].value -> ['dense0','kernel']; else None."""
    if not op_name:
        return None
    m = _STATE_PATH_RE.search(op_name)
    if not m:
        return None
    keys = _BRACKET_KEY_RE.findall(m.group(1))
    return keys or None


def _build_hierarchy(node_modpath, adj_list):
    """Turn per-node module paths (outermost first) into the nesting blobs.
    Module container ids are 'mod::dense0', 'mod::dense0/inner', etc."""
    ancestor_map = {}
    parent_module_to_nodes = defaultdict(list)
    parent_module_to_depth = {}
    module_info = {}
    node_to_module_path = {}

    def mod_id(path):
        # Must be DOT-safe (the frontend renders via Graphviz): ':' and '/' are
        # special in DOT, so sanitize like the high-level graph's container ids.
        return "mod_" + _safe_id("/".join(path))

    all_module_paths = set()
    for node_id, path in node_modpath.items():
        node_to_module_path[node_id] = "/".join(path) if path else ""
        for i in range(1, len(path) + 1):
            all_module_paths.add(tuple(path[:i]))

    # register module containers
    for path in sorted(all_module_paths, key=len):
        mid = mod_id(list(path))
        depth = len(path) - 1
        parent_module_to_depth[mid] = depth
        module_info[mid] = {"name": path[-1], "path": "/".join(path), "depth": depth}
        if len(path) == 1:
            ancestor_map[mid] = None
        else:
            ancestor_map[mid] = mod_id(list(path[:-1]))
            parent_module_to_nodes[mod_id(list(path[:-1]))].append(mid)

    # attach leaf nodes to their immediate parent module (or top level)
    for node_id, path in node_modpath.items():
        if path:
            parent = mod_id(path)
            ancestor_map[node_id] = parent
            parent_module_to_nodes[parent].append(node_id)
        else:
            ancestor_map[node_id] = None

    return (ancestor_map, dict(parent_module_to_nodes),
            parent_module_to_depth, module_info, node_to_module_path)


def _entry_block(text):
    lines = text.splitlines()
    sig_params, body, in_entry = [], [], False
    for ln in lines:
        if not in_entry and ln.lstrip().startswith("ENTRY "):
            in_entry = True
            head = ln.split("->")[0]
            sig = head[head.find("(") + 1:head.rfind(")")] if "(" in head else ""
            sig_params = _SIG_PARAM_RE.findall(sig)
            continue
        if in_entry:
            if ln.strip() == "}":
                break
            body.append(ln)
    return sig_params, body
