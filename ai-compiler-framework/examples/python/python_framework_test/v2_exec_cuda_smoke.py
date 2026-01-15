# examples/python/python_framework_test/v2_exec_cuda_smoke.py
#
# Quick smoke test for the Python-side executable above.

import os
import sys
import torch

# If you're running from repo root and not using PYTHONPATH, uncomment:
# sys.path.insert(0, os.path.abspath("build/python"))

from aicf_fw.core_v2.cuda_exec import AICFCudaExecutable


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(0)

    # Minimal JSON with already-lowered gemm
    # values: x, W, y
    ir = {
        "graph": "smoke",
        "values": {
            "v0": {"id": 0, "name": "x", "shape": [64, 8], "dtype": "torch.float32", "device": "cuda:0"},
            "v1": {"id": 1, "name": "W", "shape": [8, 8],  "dtype": "torch.float32", "device": "cuda:0"},
            "v2": {"id": 2, "name": "y", "shape": [64, 8], "dtype": "torch.float32", "device": "cuda:0"},
        },
        "nodes": [
            {"op": "gemm", "inputs": [0, 1], "outputs": [2], "attrs": {"transB": True}},
        ],
    }

    exe = AICFCudaExecutable.from_dict(ir)
    print("required_inputs:", exe.required_inputs)

    # bind required externals
    x = torch.randn(64, 8, device="cuda", dtype=torch.float32)
    W = torch.randn(8, 8, device="cuda", dtype=torch.float32)
    exe.bind_tensor("x", x)
    exe.bind_tensor("W", W)

    # y will be lazy-allocated from meta, unless you bind it
    exe.run_once()
    torch.cuda.synchronize()
    y = exe.bind["y"]
    print("run_once ok, y mean:", float(y.mean()))

    exe.capture()
    torch.cuda.synchronize()
    print("capture ok")

    exe.replay(n=5)
    torch.cuda.synchronize()
    print("replay ok")

    exe.reset_graph()
    print("reset ok")


if __name__ == "__main__":
    main()
