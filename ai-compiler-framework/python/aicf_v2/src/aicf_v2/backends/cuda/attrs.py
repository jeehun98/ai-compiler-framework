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

    if kind == "sgd_step":
        lr = float(attrs.get("lr", 1e-3))
        return struct.pack("<f", lr)
    
    if kind == "batchnorm_fwd":
        eps = float(attrs.get("eps", 1e-5))
        use_running_stats = 1 if bool(attrs.get("use_running_stats", False)) else 0
        return struct.pack("<fI", eps, use_running_stats)

    if kind == "batchnorm_bwd":
        return b""

    if kind == "layernorm_fwd":
        eps = float(attrs.get("eps", 1e-5))
        return struct.pack("<f", eps)

    if kind == "layernorm_bwd":
        return b""

    if kind == "reduce_sum":
        axis = int(attrs.get("axis", 0))
        return struct.pack("<q", axis)

    if kind == "mse_grad":
        return b""  # schema=0, default scale path

    if kind == "mse_grad_scaled":
        scale = float(attrs["scale"])  # 반드시 있어야 함
        return struct.pack("<f", scale)
    
    if kind == "relu_bwd":
        return b""  
    
    if kind == "copy":
        return b""
    
    if kind == "grad_zero":
        return b""
    
    if kind == "step_inc":
        return b""
    
    if kind == "bias_corr":
        beta1 = float(attrs.get("beta1", 0.9))
        beta2 = float(attrs.get("beta2", 0.999))
        return struct.pack("<ff", beta1, beta2)
    
    # ✅ matches your binding test: <iii> transA, transB, relu
    if kind == "gemm_epilogue":
        ta = 1 if bool(attrs.get("transA", False)) else 0
        tb = 1 if bool(attrs.get("transB", False)) else 0
        relu = 1 if bool(attrs.get("relu", True)) else 0
        return struct.pack("<iii", ta, tb, relu)

    raise KeyError(f"pack_attrs: unsupported op kind '{kind}'")