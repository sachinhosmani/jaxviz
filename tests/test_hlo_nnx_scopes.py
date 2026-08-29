import jax.numpy as jnp
import pytest

nnx = pytest.importorskip("flax.nnx")

from jaxviz import _lower_for_hlo
from jaxviz._module_scopes import parse_module_scope
from jaxviz.hlo_to_graph import (
    _dump_hlo_stages,
    _entry_block,
    _INSTR_RE,
    _OP_NAME_RE,
    _safe_id,
    _STACK_ID_RE,
    _unescape,
    build_hlo_graph,
)


class AttentionModel(nnx.Module):
    def __init__(self, rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=8,
            use_bias=False,
            decode=False,
            rngs=rngs,
        )

    def __call__(self, value):
        return self.attention(value)


class SharedLinearModel(nnx.Module):
    def __init__(self, rngs):
        self.shared = nnx.Linear(8, 8, rngs=rngs)

    def __call__(self, value):
        value = self.shared(value)
        value = nnx.relu(value)
        return self.shared(value)


def test_lowered_attention_ops_keep_their_module_path():
    model = AttentionModel(nnx.Rngs(0))
    lowered = _lower_for_hlo(model, (jnp.ones((2, 4, 8)),))
    text, _ = _dump_hlo_stages(lowered)
    _, body = _entry_block(text)
    if not any(_INSTR_RE.match(line) for line in body):
        pytest.skip("JAX version uses the legacy HLO text format")

    attention_nodes = []

    def contains_attention_scope(op_name):
        return any(
            scope is not None and scope.name == "attention"
            for provenance in op_name.split(";")
            for segment in provenance.split("/")
            for scope in (parse_module_scope(segment),)
        )

    for line in body:
        match = _INSTR_RE.match(line)
        if not match or match.group(3) == "parameter":
            continue
        op_name_match = _OP_NAME_RE.search(line)
        if op_name_match is None:
            continue
        op_name = _unescape(op_name_match.group(1))
        if (
            match.group(3) in {"reshape", "transpose"}
            and not _STACK_ID_RE.search(line)
        ):
            assert contains_attention_scope(op_name)
        if not contains_attention_scope(op_name):
            continue
        attention_nodes.append(_safe_id(match.group(1)))

    assert attention_nodes
    paths = build_hlo_graph(lowered)["node_to_module_path"]
    for node in attention_nodes:
        assert paths[node].startswith("AttentionModel/attention")


def test_lowered_reused_module_has_distinct_invocation_containers():
    model = SharedLinearModel(nnx.Rngs(1))
    lowered = _lower_for_hlo(model, (jnp.ones((2, 8)),))
    text, _ = _dump_hlo_stages(lowered)
    _, body = _entry_block(text)
    if not any(_INSTR_RE.match(line) for line in body):
        pytest.skip("JAX version uses the legacy HLO text format")

    blobs = build_hlo_graph(lowered)
    labels = blobs["graph_node_display_names"]
    shared_containers = [
        container for container in blobs["module_info"]
        if labels[container] == "shared"
    ]
    assert len(shared_containers) == 2
