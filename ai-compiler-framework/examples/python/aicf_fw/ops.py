# examples/python/aicf_fw/ops.py
import aicf_cuda

OP_ADD  = 0
OP_RELU = 1
OP_GEMM = 2

def add(x, y, out):
    aicf_cuda.op_call(OP_ADD, [x, y], [out], attrs={})
    return out

def relu(x, out):
    aicf_cuda.op_call(OP_RELU, [x], [out], attrs={})
    return out

def gemm(a, b, out, trans_a=False, trans_b=False, alpha=1.0, beta=0.0):
    aicf_cuda.op_call(OP_GEMM, [a, b], [out], attrs={
        "trans_a": bool(trans_a),
        "trans_b": bool(trans_b),
        "alpha": float(alpha),
        "beta": float(beta),
    })
    return out
