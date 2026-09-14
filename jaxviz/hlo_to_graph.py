"""Read the fully compiled HLO and build the frontend blobs.

This is the per-device view: the program as it actually runs on one device.
Everything drawn is stated by the compiler on the instruction itself --
per-device shapes, real collectives, and fusions as expandable containers.

Nothing here reconstructs the module hierarchy and nothing is recovered from an
earlier compilation stage. Every edge carries the shape its instruction states,
labelled exactly as the traced view labels its own edges. The module hierarchy
lives in that other view, built from the program as written.

Fusions are the only hierarchy. A fusion instruction names the computation it
calls, and that computation's parameters correspond positionally to the fusion's
operands, so expanding one reconnects every edge exactly. Bodies attached with
``to_apply`` (a reduction's combine function) are operator definitions rather
than dataflow and are never expanded.
"""
import glob
import os
import re
import tempfile
from collections import defaultdict

from .enums import NodeType
from .hierarchy import validate_collapsible_hierarchy

COLLECTIVE_OPCODES = {
    "all-reduce", "all-gather", "all-to-all", "reduce-scatter",
    "collective-permute", "collective-broadcast", "ragged-all-to-all",
    "all-reduce-start", "all-reduce-done", "all-gather-start",
    "all-gather-done", "collective-permute-start", "collective-permute-done",
    "reduce-scatter-start", "reduce-scatter-done",
}

# Older XLA prints instruction and computation names bare; newer XLA prefixes
# them with '%'. Both are accepted everywhere a name can appear.
_INSTR_RE = re.compile(r"^\s*(ROOT\s+)?%?([\w.\-]+)\s*=\s*(\S+)\s+([\w\-]+)\((.*)$")
_COMP_RE = re.compile(r"^(ENTRY\s+)?%?([\w.\-]+)")
_NUMERIC_RE = re.compile(r"-?\d+(\.\d+)?([eE][-+]?\d+)?")
_OP_NAME_RE = re.compile(r'op_name="((?:[^"\\]|\\.)*)"')
_CALLS_RE = re.compile(r"\bcalls=(%?[\w.\-]+(?:\s*,\s*%?[\w.\-]+)*)")
_SHARDING_RE = re.compile(r"sharding=(\{[^}]*\})")

# state['dense0']['kernel'].value  and  state["dense0"]["kernel"].value
_STATE_PATH_RE = re.compile(r"""(?:state|params)((?:\[\s*['"][^'"]+['"]\s*\])+)""")
_BRACKET_KEY_RE = re.compile(r"""\[\s*['"]([^'"]+)['"]\s*\]""")

# Not call parameters -- placement/metadata, excluded from keyword_args.
_ATTR_EXCLUDE = {"metadata", "sharding", "backend_config", "frontend_attributes"}


def _safe_id(s):
    return re.sub(r"[^0-9A-Za-z_]", "_", s)


def _unescape(s):
    return s.replace("\\'", "'").replace('\\"', '"') if s else s


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


# --------------------------------------------------------------------------
# Dump: the fully compiled module
# --------------------------------------------------------------------------
def _dump_compiled_hlo(lowered):
    """Compile with dumping on and return the final compiled module's text.

    One stage, the same for every program: the end of the compiler's pipeline,
    the code that actually runs. Nothing is read from any earlier stage, so
    nothing has to be matched between stages.
    """
    dump_dir = tempfile.mkdtemp(prefix="jaxviz_hlo_")
    lowered.compile(compiler_options={
        "xla_dump_to": dump_dir,
        "xla_dump_hlo_as_text": True,
    })

    modules = [
        path for path in sorted(glob.glob(os.path.join(dump_dir, "*.txt")))
        if os.path.basename(path).endswith("after_optimizations.txt")
    ]
    if not modules:
        raise RuntimeError(f"No compiled HLO module produced in {dump_dir}")

    # Several modules are emitted; the traced function is the largest. The rest
    # are helpers compiled for device_put and friends.
    return open(max(modules, key=os.path.getsize)).read()


# --------------------------------------------------------------------------
# Module parsing
# --------------------------------------------------------------------------
class _Instruction:
    __slots__ = ("name", "shape", "opcode", "operands", "tail", "op_name",
                 "sharding", "calls", "param_index", "is_root")

    def __init__(self, name, shape, opcode, operands, tail, is_root):
        self.name = name
        self.shape = shape
        self.opcode = opcode
        self.operands = operands
        self.tail = tail
        self.is_root = is_root
        on = _OP_NAME_RE.search(tail)
        self.op_name = _unescape(on.group(1)) if on else None
        sh = _SHARDING_RE.search(tail)
        self.sharding = sh.group(1) if sh else None
        calls = _CALLS_RE.search(tail)
        self.calls = ([c.strip().lstrip("%") for c in calls.group(1).split(",")]
                      if calls else [])
        self.param_index = None


class _Computation:
    __slots__ = ("name", "is_entry", "instructions", "root")

    def __init__(self, name, is_entry):
        self.name = name
        self.is_entry = is_entry
        self.instructions = []
        self.root = None


def _parse_operands(operand_str):
    """Operand names from the text between an opcode's parentheses.

    Names may or may not carry a '%'. Bare numbers are literal arguments
    (``parameter(0)``, ``constant(2)``), never operands.
    """
    operands = []
    for token in _split_top_commas(operand_str):
        token = token.strip().lstrip("%")
        if not token or _NUMERIC_RE.fullmatch(token):
            continue
        if re.fullmatch(r"[\w.\-]+", token):
            operands.append(token)
    return operands


def _parse_module(text):
    """Split the module text into computations and their instructions."""
    computations = {}
    current = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if current is None:
            if stripped.endswith("{") and not stripped.startswith("HloModule"):
                m = _COMP_RE.match(stripped)
                if not m:
                    continue
                current = _Computation(m.group(2), bool(m.group(1)))
            continue

        if stripped == "}":
            computations[current.name] = current
            current = None
            continue

        m = _INSTR_RE.match(raw)
        if not m:
            continue
        is_root, name, shape, opcode, rest = m.groups()
        operand_str, _, tail = rest.partition(")")
        operands = _parse_operands(operand_str)
        instruction = _Instruction(name, shape, opcode, operands, tail,
                                   bool(is_root))
        if opcode == "parameter":
            index = re.search(r"\d+", operand_str)
            instruction.param_index = int(index.group(0)) if index else 0
            instruction.operands = []
        current.instructions.append(instruction)
        if instruction.is_root:
            current.root = instruction.name

    if current is not None:               # file ended without a closing line
        computations[current.name] = current

    entry = next((c for c in computations.values() if c.is_entry), None)
    if entry is None:
        raise RuntimeError("No ENTRY computation found in the HLO dump")
    return computations, entry


def _state_path(op_name):
    """state['dense0']['kernel'].value -> ['dense0','kernel']; else None."""
    if not op_name:
        return None
    m = _STATE_PATH_RE.search(op_name)
    if not m:
        return None
    return _BRACKET_KEY_RE.findall(m.group(1)) or None


def _split_top_commas(s):
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


def _parse_hlo_attrs(tail):
    """The op's genuine static parameters, excluding placement/metadata."""
    attrs = {}
    for piece in _split_top_commas(tail):
        piece = piece.strip().strip(",").strip()
        if "=" not in piece:
            continue
        key, val = piece.split("=", 1)
        key = key.strip()
        if key in _ATTR_EXCLUDE or not re.fullmatch(r"[a-z_][a-z0-9_]*", key):
            continue
        attrs[key] = val.strip()[:120]
    return attrs


def _label_for(instruction):
    """A fusion is named for the operation it was built from when the compiler
    says so; everything else is named by its opcode."""
    if instruction.op_name and instruction.opcode in ("fusion", "call"):
        # Older XLA appends the primitive's static params: 'dot_general[...]'.
        tail = instruction.op_name.split("/")[-1].split("[")[0].strip()
        if tail:
            return tail
    return instruction.opcode


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_hlo_graph(lowered):
    text = _dump_compiled_hlo(lowered)
    computations, entry = _parse_module(text)

    adj_list = {}
    func_info = {}
    module_info = {}
    graph_node_display_names = {}
    graph_node_name_to_without_suffix = {}
    node_to_module_path = {}
    node_to_attr_name = {}
    ancestor_map = {}
    parent_module_to_nodes = defaultdict(list)
    parent_module_to_depth = {}

    counter = [0]
    out_shape = {}      # node id -> raw shape string
    seen_edges = set()

    def new_id(base):
        counter[0] += 1
        return f"{_safe_id(base)}_{counter[0]}"

    def add_node(node_id, node_type, label, container):
        adj_list[node_id] = {"edges": [], "failed": False, "node_type": node_type}
        graph_node_display_names[node_id] = label
        graph_node_name_to_without_suffix[node_id] = label
        ancestor_map[node_id] = container
        if container is not None:
            parent_module_to_nodes[container].append(node_id)

    def add_edge(src, dst):
        if src is None or src not in adj_list or src == dst:
            return
        key = (src, dst)
        if key in seen_edges:
            return
        seen_edges.add(key)
        adj_list[src]["edges"].append({
            "target": dst,
            "dims": _hlo_shape_to_dims(out_shape.get(src, "")),
            "edge_data_id": src,
        })

    def emit(computation, param_sources, container, depth):
        """Emit a computation's instructions. ``param_sources`` maps a parameter
        index to the node producing that operand in the calling scope. Returns
        the node producing this computation's result."""
        local = {}

        def producer(operand_name):
            return local.get(operand_name)

        for instruction in computation.instructions:
            if instruction.opcode == "parameter" and not computation.is_entry:
                # Wiring only: the value comes from the caller's operand.
                local[instruction.name] = param_sources.get(instruction.param_index)
                continue

            sources = [producer(o) for o in instruction.operands]

            if instruction.opcode == "parameter":
                node_id = new_id(instruction.name)
                path = _state_path(instruction.op_name)
                if path is not None:
                    add_node(node_id, NodeType.PARAMETER.value,
                             ".".join(path), container)
                    graph_node_name_to_without_suffix[node_id] = "param"
                    node_to_attr_name[node_id] = path[-1]
                elif instruction.op_name and "/" not in instruction.op_name \
                        and not instruction.op_name.startswith("jit("):
                    add_node(node_id, NodeType.INPUT.value,
                             instruction.op_name, container)
                    graph_node_name_to_without_suffix[node_id] = "input"
                else:
                    add_node(node_id, NodeType.PARAMETER.value, "param", container)
                func_info[node_id] = {"positional_args": [], "keyword_args": {}}
                out_shape[node_id] = instruction.shape
                local[instruction.name] = node_id
                continue

            called = [c for c in instruction.calls if c in computations]
            if len(called) == 1:
                # An expandable container: its body is dataflow, and its
                # parameters line up positionally with these operands.
                container_id = new_id(instruction.name)
                label = _label_for(instruction)
                graph_node_display_names[container_id] = label
                graph_node_name_to_without_suffix[container_id] = label
                module_info[container_id] = {
                    "type": label,
                    "parameters": {},
                    "attributes": _parse_hlo_attrs(instruction.tail),
                }
                ancestor_map[container_id] = container
                parent_module_to_depth[container_id] = depth
                if container is not None:
                    parent_module_to_nodes[container].append(container_id)
                node_to_module_path[container_id] = instruction.op_name or ""

                inner_root = emit(
                    computations[called[0]],
                    {i: s for i, s in enumerate(sources)},
                    container_id,
                    depth + 1,
                )
                local[instruction.name] = inner_root
                continue

            # An ordinary instruction.
            node_id = new_id(instruction.name)
            is_collective = instruction.opcode in COLLECTIVE_OPCODES
            node_type = (NodeType.COLLECTIVE.value if is_collective
                         else NodeType.CONSTANT.value
                         if instruction.opcode == "constant"
                         else NodeType.OPERATION.value)
            add_node(node_id, node_type, instruction.opcode, container)
            out_shape[node_id] = instruction.shape
            node_to_module_path[node_id] = instruction.op_name or ""

            keyword_args = _parse_hlo_attrs(instruction.tail)
            if instruction.op_name:
                keyword_args["from"] = instruction.op_name
            func_info[node_id] = {
                "positional_args": [
                    {"_type": "tensor",
                     "shape": _hlo_shape_parts(out_shape.get(s, ""))[1],
                     "dtype": _hlo_shape_parts(out_shape.get(s, ""))[0]}
                    for s in sources if s is not None
                ],
                "keyword_args": keyword_args,
            }

            for src in sources:
                add_edge(src, node_id)
            local[instruction.name] = node_id

        return local.get(computation.root)

    root_node = emit(entry, {}, None, 0)

    if root_node is not None:
        add_node("output_0", NodeType.OUTPUT.value, "output_0", None)
        graph_node_name_to_without_suffix["output_0"] = "output"
        add_edge(root_node, "output_0")

    validate_collapsible_hierarchy(adj_list, ancestor_map, module_info)

    return {
        "adj_list": adj_list,
        "module_info": module_info,
        "func_info": func_info,
        "node_to_module_path": node_to_module_path,
        "parent_module_to_nodes": dict(parent_module_to_nodes),
        "parent_module_to_depth": parent_module_to_depth,
        "graph_node_name_to_without_suffix": graph_node_name_to_without_suffix,
        "graph_node_display_names": graph_node_display_names,
        "node_to_attr_name": node_to_attr_name,
        "ancestor_map": ancestor_map,
        "repeat_containers": [],
    }
