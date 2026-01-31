import struct

def pack_attrs(kind, attrs, *, runtime_flags=None) -> bytes:
    runtime_flags = runtime_flags or {}

    if kind == "gemm":
        ta = 1 if bool(attrs.get("transA", False)) else 0
        tb = 1 if bool(attrs.get("transB", False)) else 0
        return struct.pack("<ii", ta, tb)

    if kind == "bias_add":
        axis = int(attrs.get("broadcast_axis", -1))
        return struct.pack("<q", axis)

    if kind in ("relu", "add"):
        return b""

    if kind == "adam_step":
        # schema: 'ADAM'
        # blob: <ffff = lr, beta1, beta2, eps
        lr = float(attrs.get("lr", 1e-3))
        beta1 = float(attrs.get("beta1", 0.9))
        beta2 = float(attrs.get("beta2", 0.999))
        eps = float(attrs.get("eps", 1e-8))
        return struct.pack("<ffff", lr, beta1, beta2, eps)

    raise KeyError(f"pack_attrs: unsupported op kind '{kind}'")
