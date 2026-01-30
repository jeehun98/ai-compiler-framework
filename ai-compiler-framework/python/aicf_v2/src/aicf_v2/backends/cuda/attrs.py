import struct

def pack_attrs(kind, attrs, *, runtime_flags=None) -> bytes:
    if kind == "gemm":
        ta = 1 if bool(attrs.get("transA", False)) else 0
        tb = 1 if bool(attrs.get("transB", False)) else 0
        return struct.pack("<ii", ta, tb)

    if kind == "bias_add":
        axis = int(attrs.get("broadcast_axis", -1))
        return struct.pack("<q", axis)

    if kind in ("relu", "add"):
        return b""

    raise KeyError(...)
