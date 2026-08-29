import base64
import binascii
from dataclasses import dataclass


MODULE_SCOPE_PREFIX = "__jaxviz_module__"


@dataclass(frozen=True)
class ModuleInvocation:
    name: str
    invocation: int

    @property
    def identity(self):
        encoded_name = _encode_name(self.name)
        return f"{self.invocation}_{encoded_name}"


def _encode_name(name):
    encoded = base64.urlsafe_b64encode(name.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def _decode_name(encoded):
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode(encoded + padding).decode("utf-8")


def module_scope_name(name, invocation):
    scope = ModuleInvocation(str(name), int(invocation))
    return MODULE_SCOPE_PREFIX + scope.identity


def parse_module_scope(segment):
    if not segment.startswith(MODULE_SCOPE_PREFIX):
        return None
    payload = segment[len(MODULE_SCOPE_PREFIX):]
    invocation_text, separator, encoded_name = payload.partition("_")
    if not separator or not invocation_text.isdigit() or not encoded_name:
        return None
    try:
        name = _decode_name(encoded_name)
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    return ModuleInvocation(name, int(invocation_text))


def module_path_from_name_stack(name_stack):
    stack = getattr(name_stack, "stack", ())
    if stack:
        segments = [getattr(entry, "name", "") for entry in stack]
    else:
        segments = str(name_stack or "").split("/")
    return [
        scope
        for segment in segments
        if (scope := parse_module_scope(segment)) is not None
    ]
