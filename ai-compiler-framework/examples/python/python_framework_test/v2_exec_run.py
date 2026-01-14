from __future__ import annotations

import sys
from pathlib import Path
import torch

# ------------------------------------------------------------
# Path bootstrap
# ------------------------------------------------------------
THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

from aicf_fw.backend.aicf_backend import AICFBackend
from aicf_fw.backend import set_backend

from aicf_fw.core_v2 import (
    trace_ir,
    dump_ir,
    lower_to_backend_ops,
    dump_lowered,
    build_binding_plan,
    dump_plan,
    PlannedExecutor,
    ExecOptions,
)
from aicf_fw.core_v2 import sym_tensor, linear, relu, mse_grad


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    dev = torch.device("cuda")
    dt = torch.float32

    # backend set
    bk = AICFBackend()
    set_backend(bk)

    # --- symbolic graph (trace only) ---
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

    ir = trace_ir(step_fn, name="v2_stage4")
    lowered = lower_to_backend_ops(ir)
    plan = build_binding_plan(ir)

    print(dump_ir(ir))
    print("")
    print(dump_lowered(lowered, title="LoweredOps(v2_stage4)"))
    print("")
    print(dump_plan(plan, title="BindingPlan(v2_stage4)"))

    # --- runtime tensors (real) ---
    x_rt = torch.randn(64, 8, device=dev, dtype=dt)
    t_rt = torch.randn(64, 8, device=dev, dtype=dt)

    W0_rt = torch.randn(8, 8, device=dev, dtype=dt)
    b0_rt = torch.randn(8, device=dev, dtype=dt)
    W1_rt = torch.randn(8, 8, device=dev, dtype=dt)
    b1_rt = torch.randn(8, device=dev, dtype=dt)

    # executor
    ex = PlannedExecutor(
        ir=ir,
        lowered=lowered,
        plan=plan,
        backend=bk,
        device=dev,
        opts=ExecOptions(debug=True, debug_limit=20),
    )

    env = ex.run(
        inputs={"x": x_rt, "t": t_rt},
        params={"0.W": W0_rt, "0.b": b0_rt, "2.W": W1_rt, "2.b": b1_rt},
        reuse_static=False,
    )
    torch.cuda.synchronize()

    # check output dY (vid by name)
    # find vid of dY from plan
    dY_vid = None
    for vid, spec in plan.specs.items():
        if spec.name == "dY":
            dY_vid = int(vid)
            break
    assert dY_vid is not None
    dY = env[dY_vid]

    # compare with torch reference for sanity
    with torch.no_grad():
        y0_ref = x_rt @ W0_rt.t() + b0_rt
        a0_ref = torch.relu(y0_ref)
        y1_ref = a0_ref @ W1_rt.t() + b1_rt
        dY_ref = 2.0 * (y1_ref - t_rt) / (y1_ref.numel())

        maxdiff = (dY - dY_ref).abs().max().item()
        print(f"[check] dY maxabs diff vs torch = {maxdiff:.6e}")

    print("OK")


if __name__ == "__main__":
    main()
