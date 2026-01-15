import sys
import os

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../build/python")
)
sys.path.insert(0, ROOT)

import torch
import aicf_cuda
from aicf_cuda import _C

def main():
    torch.manual_seed(0)
    assert torch.cuda.is_available()

    # ---- op_call smoke (current stream) ----
    x = torch.randn(64, 8, device="cuda", dtype=torch.float32)
    W = torch.randn(8, 8, device="cuda", dtype=torch.float32)
    y = torch.empty(64, 8, device="cuda", dtype=torch.float32)

    _C.trace_reset()
    _C.op_call(int(_C.OpKind.Gemm), [x, W], [y], {"transB": True})
    torch.cuda.synchronize()
    print("op_call gemm ok, trace:", _C.trace_get())

    # ---- graph capture smoke (dedicated stream) ----
    # capture stream handle
    s = _C.graph_begin()
    _C.trace_reset()

    # IMPORTANT: pass stream=s
    _C.op_call(int(_C.OpKind.Gemm), [x, W], [y], {"transB": True}, stream=s)

    _C.graph_end()
    torch.cuda.synchronize()
    print("graph captured, trace:", _C.trace_get())

    # replay a few times
    for i in range(5):
        _C.graph_launch()
    torch.cuda.synchronize()
    print("graph_launch ok")

if __name__ == "__main__":
    main()
