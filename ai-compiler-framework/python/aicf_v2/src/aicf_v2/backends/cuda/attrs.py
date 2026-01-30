from __future__ import annotations
import struct
from typing import Any, Dict

def pack_attrs(kind: str, attrs: Dict[str, Any], *, runtime_flags=None) -> bytes:
    if kind == "bias_add":
        # ✅ BiasAddAttrV0: int64 axis
        axis = int(attrs.get("broadcast_axis", -1))
        return struct.pack("<q", axis)

    if kind == "gemm":
        # 아직 C++ struct 모르면 일단 빈 바이트로 두고
        # 다음 에러(잘못된 attr size/schema)를 보고 맞추는 게 빠름
        return b""

    if kind in ("relu", "add"):
        return b""

    raise KeyError(f"pack_attrs: unsupported kind '{kind}'")
