from __future__ import annotations

import sys
from pathlib import Path

# ------------------------------------------------------------
# Path bootstrap
# ------------------------------------------------------------
THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

import torch

from aicf_fw.core_v2 import (
    trace_ir,
    dump_ir,
    lower_to_backend_ops,
    dump_lowered,
    build_binding_plan,
    dump_plan,
    allocate_static_env,
)
from aicf_fw.core_v2 import sym_tensor, linear, relu, mse_grad


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt = torch.float32

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
        _ = mse_grad(y1, t, name="dY")

    ir = trace_ir(step_fn, name="v2_stage3")
    lowered = lower_to_backend_ops(ir)
    plan = build_binding_plan(ir)

    print(dump_ir(ir))
    print("")
    print(dump_lowered(lowered, title="LoweredOps(v2_stage3)"))
    print("")
    print(dump_plan(plan, title="BindingPlan(v2_stage3)"))

    # optional: show static env allocation summary
    env = allocate_static_env(ir, plan, device=dev)
    print("=== StaticEnvAlloc ===")
    print("allocated:", len(env))
    for vid in sorted(env.keys()):
        tt = env[vid]
        print(f"  v{vid:03d} ptr={tt.data_ptr()} shape={tuple(tt.shape)} dtype={tt.dtype} dev={tt.device}")


if __name__ == "__main__":
    main()
