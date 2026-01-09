from __future__ import annotations

import os
import sys
from pathlib import Path
import random
import numpy as np
import torch

THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

REPLAY_N = int(os.environ.get("AICF_REPLAY_N", "20"))
SEED = int(os.environ.get("AICF_SEED", "0"))
WARMUP_RUNS = int(os.environ.get("AICF_WARMUP_RUNS", "2"))

IR_DUMP = int(os.environ.get("AICF_IR_DUMP", "1"))
LOWER_DUMP = int(os.environ.get("AICF_LOWER_DUMP", "1"))
TRACE_DUMP = int(os.environ.get("AICF_TRACE_DUMP", "1"))
TRACE_FILTER = int(os.environ.get("AICF_TRACE_FILTER", "1"))
CHECK_RESTORE = int(os.environ.get("AICF_CHECK_RESTORE", "1"))

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ok(msg: str): print(f"[OK] {msg}")
def fail(msg: str): raise SystemExit(f"[FAIL] {msg}")

def main():
    print("=== PR3 verify (TrainGraph API): static inputs + capture/replay ===")
    if not torch.cuda.is_available():
        fail("CUDA not available")
    seed_all(SEED)

    from aicf_fw.backend.aicf_backend import AICFBackend
    from aicf_fw.backend import set_backend
    from aicf_fw.nn.linear import Linear
    from aicf_fw.nn.relu import ReLU
    from aicf_fw.nn.sequential import Sequential
    from aicf_fw.optim.adam import Adam

    os.environ.setdefault("AICF_ENFORCE_ADAMSTEP_RUNTIME", "1")

    set_backend(AICFBackend())

    model = Sequential(
        Linear(8, 8, device="cuda", dtype=torch.float32),
        ReLU(),
        Linear(8, 8, device="cuda", dtype=torch.float32),
    )
    optim = Adam(model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, grad_clip=None)

    tg = model.compile_train(
        optim=optim,
        input_spec={
            "x": ((64, 8), torch.float32, "cuda"),
            "t": ((64, 8), torch.float32, "cuda"),
        },
        loss="mse",
        name="train_step_static_io",
        warmup_runs=WARMUP_RUNS,
        validate=True,
        trace=True,
        enforce_ops=("adam_step",),
    )

    # feed once before validations (ensures non-zero grads in practice)
    x0 = torch.randn(64, 8, device="cuda", dtype=torch.float32)
    t0 = torch.randn(64, 8, device="cuda", dtype=torch.float32)
    tg.set_inputs(x=x0, t=t0)
    
    if IR_DUMP or LOWER_DUMP or TRACE_DUMP:
        tg.dump(ir=bool(IR_DUMP), lowered=bool(LOWER_DUMP), trace=bool(TRACE_DUMP))

    tg.assert_runtime_matches_lowering(trace_filter=bool(TRACE_FILTER))
    ok("runtime trace matches lowering")

    tg.assert_adam_state_mutates(tag="smoke")
    ok("adam mutates on replay")

    tg.assert_determinism(replays=REPLAY_N, check_restore=bool(CHECK_RESTORE))
    ok("determinism OK")

    # 실제 사용 예: 입력을 바꿔가며 replay
    for i in range(3):
        x = torch.randn(64, 8, device="cuda", dtype=torch.float32)
        t = torch.randn(64, 8, device="cuda", dtype=torch.float32)
        tg.set_inputs(x=x, t=t)
        tg.replay()
    torch.cuda.synchronize()
    ok("dynamic feeding via static buffers OK")

    print("OK.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
