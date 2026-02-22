# python/aicf_v2/tests/test_dump_and_timing_mnist.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
from pathlib import Path as _Path

# -----------------------
# project path bootstrap
# -----------------------
p = _Path(__file__).resolve()
root = next(parent for parent in [p] + list(p.parents) if (parent / "pyproject.toml").exists())
build_lib_path = root / "build" / "python" / "aicf_cuda"
src_path = root / "python" / "aicf_v2" / "src"
if build_lib_path.exists():
    sys.path.insert(0, str(build_lib_path))
sys.path.insert(0, str(src_path))

import aicf_v2 as aicf
from aicf_v2.optimizers.adam import Adam


# -----------------------
# dump utilities (robust)
# -----------------------
def _safe(obj: Any) -> Any:
    """json-serializable로 최대한 변환"""
    if obj is None:
        return None
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes_len__": len(obj)}
    if hasattr(obj, "__class__") and obj.__class__.__name__ in ("dtype", "device"):
        return str(obj)
    return repr(obj)


def dump_builder(model: Any, out_dir: Path, stage: str) -> Path:
    b = model.b
    out_dir.mkdir(parents=True, exist_ok=True)

    values_obj = getattr(b, "values", None)
    if values_obj is None:
        raise RuntimeError("Builder has no attribute 'values'")

    values: Dict[str, Any] = {}
    if isinstance(values_obj, list):
        for vid, v in enumerate(values_obj):
            if v is None:
                continue
            spec = getattr(v, "spec", None)
            values[str(vid)] = {
                "vid": int(getattr(v, "vid", vid)),
                "name": getattr(v, "name", None),
                "kind": getattr(v, "kind", None),
                "spec": None if spec is None else {
                    "shape": list(getattr(spec, "shape", ())),
                    "dtype": getattr(spec, "dtype", None),
                    "device": getattr(spec, "device", None),
                },
            }
    elif isinstance(values_obj, dict):
        for vid, v in values_obj.items():
            if v is None:
                continue
            spec = getattr(v, "spec", None)
            values[str(int(vid))] = {
                "vid": int(getattr(v, "vid", vid)),
                "name": getattr(v, "name", None),
                "kind": getattr(v, "kind", None),
                "spec": None if spec is None else {
                    "shape": list(getattr(spec, "shape", ())),
                    "dtype": getattr(spec, "dtype", None),
                    "device": getattr(spec, "device", None),
                },
            }
    else:
        raise TypeError(f"Unsupported Builder.values type: {type(values_obj)}")

    ops: List[Dict[str, Any]] = []
    for i, op in enumerate(getattr(b, "ops", [])):
        ops.append({
            "i": i,
            "kind": getattr(op, "kind", None),
            "name": getattr(op, "name", None),
            "inputs": [int(x) for x in getattr(op, "inputs", [])],
            "outputs": [int(x) for x in getattr(op, "outputs", [])],
            "hints": _safe(getattr(op, "hints", None)),
            "attr_schema": _safe(getattr(op, "attr_schema", None)),
            "attr_blob_len": None if getattr(op, "attr_blob", None) is None else len(getattr(op, "attr_blob")),
        })

    payload = {
        "stage": stage,
        "values_container_type": type(values_obj).__name__,
        "num_values": len(values_obj),
        "num_ops": len(getattr(b, "ops", [])),
        "input_vids": [int(x) for x in getattr(b, "input_vids", [])],
        "param_vids": [int(x) for x in getattr(b, "param_vids", [])],
        "state_vids": [int(x) for x in getattr(b, "state_vids", [])],
        "external_vids": [int(x) for x in getattr(b, "external_vids", [])],
        "output_map": _safe(getattr(b, "outputs", None)),
        "values": values,
        "ops": ops,
    }

    path = out_dir / f"{stage}__builder.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[DUMP] {path}")
    return path


def dump_plan(model: Any, compiled_program: Any, out_dir: Path, stage: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = getattr(compiled_program, "plan", None)

    if plan is None:
        payload = {"stage": stage, "error": "compiled_program.plan is None"}
    else:
        plan_ops = []
        for i, op in enumerate(getattr(plan, "ops", [])):
            plan_ops.append({
                "i": i,
                "kind": getattr(op, "kind", None),
                "name": getattr(op, "name", None),
                "inputs": [int(x) for x in getattr(op, "inputs", [])],
                "outputs": [int(x) for x in getattr(op, "outputs", [])],
                "kind_id": _safe(getattr(op, "kind_id", None)),
                "attr_schema": _safe(getattr(op, "attr_schema", None)),
                "attr_blob_len": None if getattr(op, "attr_blob", None) is None else len(getattr(op, "attr_blob")),
                "hints": _safe(getattr(op, "hints", None)),
            })

        payload = {
            "stage": stage,
            "plan_id": _safe(getattr(plan, "plan_id", None)),
            "num_ops": len(getattr(plan, "ops", [])),
            "alias": {str(int(k)): int(v) for k, v in getattr(plan, "alias", {}).items()},
            "ops": plan_ops,
        }

    path = out_dir / f"{stage}__plan.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[DUMP] {path}")
    return path


# -----------------------
# timing utilities
# -----------------------
@torch.no_grad()
def time_steps_cuda(
    fn,
    steps: int,
    *,
    label: str,
    warmup: int = 5,
) -> Tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(steps):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = float(start.elapsed_time(end))
    avg_ms = total_ms / steps
    print(f"[TIME] {label}: total={total_ms:.3f} ms  avg={avg_ms:.3f} ms/step  (steps={steps})")
    return avg_ms, total_ms


def time_wall(fn, steps: int, *, label: str, warmup: int = 2) -> Tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_s = t1 - t0
    avg_ms = (total_s * 1000.0) / steps
    print(f"[WALL] {label}: total={total_s:.6f} s  avg={avg_ms:.3f} ms/step  (steps={steps})")
    return avg_ms, total_s


# -----------------------
# main test
# -----------------------
def main():
    device = "cuda"
    torch.backends.cudnn.benchmark = True

    batch_size = 64
    lr = 1e-3
    beta1, beta2 = 0.9, 0.999

    dump_dir = Path("./aicf_dumps_mnist")
    dump_dir.mkdir(parents=True, exist_ok=True)

    # 1) data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    # 2) model graph build (FWD)
    model = aicf.Sequential([
        aicf.Linear(784, 128, name="fc1"),
        aicf.ReLU(name="relu1"),
        aicf.Linear(128, 10, name="fc2"),
    ])

    x_spec = aicf.TensorSpec(shape=(batch_size, 784), dtype="f32", device="cuda")
    y_pred_vid = model.build(x_spec, input_name="x")
    _ = y_true_vid = model.input("y_true", aicf.TensorSpec(shape=(batch_size,), dtype="i32", device="cuda"))

    loss_vid = model.add(aicf.CrossEntropyLoss(reduction="mean", name="loss"), y_pred_vid, y_true_vid)

    model.b.outputs["prob"] = y_pred_vid
    model.b.outputs["loss"] = loss_vid

    dump_builder(model, dump_dir, "01_fwd_built")

    # 3) (NEW) fwd_opt snapshot -> bwd graph build
    # optimize가 아직 identity여도, 논리 구조를 먼저 확정
    model.build_backward_after_fwd_opt(loss_vid)

    optimizer = Adam(model, lr=lr)
    optimizer.step()

    dump_builder(model, dump_dir, "02_bwd_built_post_fwd_opt")

    # 4) static buffers (주소 고정)
    static_x = torch.zeros((batch_size, 784), device=device, dtype=torch.float32).contiguous()
    static_y = torch.zeros((batch_size,), device=device, dtype=torch.int32).contiguous()
    static_grad_init = torch.ones((1,), device=device, dtype=torch.float32).contiguous()

    # bias correction도 static으로 고정
    static_bc1 = torch.zeros((1,), device=device, dtype=torch.float32).contiguous()
    static_bc2 = torch.zeros((1,), device=device, dtype=torch.float32).contiguous()

    # 5) compile (no capture) + dump plan
    step = 1
    static_bc1.fill_(1.0 - (beta1 ** step))
    static_bc2.fill_(1.0 - (beta2 ** step))

    sample_feed = {
        "x": static_x,
        "y_true": static_y,
        "grad_initial": static_grad_init,
        "adam.bc1": static_bc1,
        "adam.bc2": static_bc2,
    }

    model.compile(capture=False, sample_feed=sample_feed, mode="train")
    dump_plan(model, model.compiled_program, dump_dir, "03_compiled_plan")

    # 6) capture
    print("[TEST] Capturing CUDA Graph ...")
    model.compile(capture=True, sample_feed=sample_feed, mode="train")
    print("[TEST] Capture done.")

    # 7) prepare one batch source (고정된 1개 배치로 시간 측정)
    data0, target0 = next(iter(train_loader))
    data0 = data0.view(batch_size, -1).to(device, non_blocking=True)
    target0 = target0.to(torch.int32).to(device, non_blocking=True)

    step_counter = {"step": 0}

    def _prepare_feed_for_step() -> Dict[str, torch.Tensor]:
        step_counter["step"] += 1
        s = step_counter["step"]

        static_x.copy_(data0)
        static_y.copy_(target0)

        static_bc1.fill_(1.0 - (beta1 ** s))
        static_bc2.fill_(1.0 - (beta2 ** s))

        return {
            "x": static_x,
            "y_true": static_y,
            "grad_initial": static_grad_init,
            "adam.bc1": static_bc1,
            "adam.bc2": static_bc2,
        }

    # 8) timing: eager vs cuda graph
    steps = 200

    def step_eager():
        feed = _prepare_feed_for_step()
        _ = model.run(feed, use_cuda_graph=False, mode="train")

    def step_graph():
        feed = _prepare_feed_for_step()
        _ = model.run(feed, use_cuda_graph=True, mode="train")

    print("\n===== TIMING (CUDA events: kernel-side) =====")
    step_counter["step"] = 0
    time_steps_cuda(step_eager, steps, label="Eager (no CUDA Graph)", warmup=10)

    step_counter["step"] = 0
    time_steps_cuda(step_graph, steps, label="CUDA Graph replay", warmup=10)

    print("\n===== TIMING (Wall: python+sync included) =====")
    step_counter["step"] = 0
    time_wall(step_eager, steps, label="Eager (no CUDA Graph)", warmup=5)

    step_counter["step"] = 0
    time_wall(step_graph, steps, label="CUDA Graph replay", warmup=5)

    # 9) quick correctness sanity
    step_counter["step"] = 0
    out = model.run(_prepare_feed_for_step(), use_cuda_graph=True, mode="train")
    print(f"\n[SANITY] loss={out['loss'].item():.6f}  prob.shape={tuple(out['prob'].shape)}")

    print(f"\n[DONE] dumps in: {dump_dir.resolve()}")


if __name__ == "__main__":
    main()