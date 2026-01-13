from __future__ import annotations

import sys
from pathlib import Path

# ------------------------------------------------------------
# Path bootstrap (match your minitest style)
# ------------------------------------------------------------
THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))


import torch

from aicf_fw.core_v2 import trace_ir, dump_ir
from aicf_fw.core_v2 import sym_tensor, linear, relu, mse_grad


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float32

    # 심볼릭 입력/파라미터
    x = sym_tensor(name="x", shape=(64, 8), dtype=dt, device=dev)
    t = sym_tensor(name="t", shape=(64, 8), dtype=dt, device=dev)

    W0 = sym_tensor(name="0.W", shape=(8, 8), dtype=dt, device=dev)
    b0 = sym_tensor(name="0.b", shape=(8,), dtype=dt, device=dev)
    W1 = sym_tensor(name="2.W", shape=(8, 8), dtype=dt, device=dev)
    b1 = sym_tensor(name="2.b", shape=(8,), dtype=dt, device=dev)

    def step_fn():
        y0 = linear(x, W0, b0, name="lin0_out")
        a0 = relu(y0, name="relu0_out")
        y1 = linear(a0, W1, b1, name="lin1_out")
        _ = mse_grad(y1, t, name="dY")  # 1단계에선 backward/optim은 아직 안 함

    ir = trace_ir(step_fn, name="v2_stage1")
    print(dump_ir(ir))


if __name__ == "__main__":
    main()
