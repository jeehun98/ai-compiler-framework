from __future__ import annotations
import struct
from typing import Any, Dict

def pack_attrs(kind: str, attrs: Dict[str, Any], *, runtime_flags=None) -> bytes:
    if kind == "gemm":
        ta = 1 if bool(attrs.get("transA", False)) else 0
        tb = 1 if bool(attrs.get("transB", False)) else 0
        return struct.pack("<ii", int(ta), int(tb))

    if kind == "bias_add":
        axis = int(attrs.get("broadcast_axis", -1))
        return struct.pack("<q", axis)

    if kind in ("relu", "add"):
        return b""

    raise KeyError(f"pack_attrs: unsupported kind '{kind}'")
