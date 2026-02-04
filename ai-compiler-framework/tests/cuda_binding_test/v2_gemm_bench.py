# cuda_binding_test/v2_gemm_bench.py
from __future__ import annotations

import os
import sys
import json
import time
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]
EX_PY = ROOT / "examples" / "python"
BUILD_PY = ROOT / "build" / "python"
for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import _C  # noqa: E402


# ----------------------------
# Attr pack
# ----------------------------
def pack_gemm(trans_a: int = 0, trans_b: int = 0) -> bytes:
    return struct.pack("<ii", int(trans_a), int(trans_b))


# ----------------------------
# Artifacts
# ----------------------------
@dataclass
class ArtifactRun:
    tag: str
    out_dir: Path

    @staticmethod
    def create(tag: str) -> "ArtifactRun":
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "artifacts" / f"{ts}_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return ArtifactRun(tag=tag, out_dir=out_dir)

    def write_text(self, name: str, s: str) -> None:
        (self.out_dir / name).write_text(s, encoding="utf-8")

    def write_json(self, name: str, obj: Any) -> None:
        (self.out_dir / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def dump_env() -> str:
    lines = []
    lines.append(f"python: {sys.version}")
    lines.append(f"torch: {torch.__version__}")
    lines.append(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        lines.append(f"device_count: {torch.cuda.device_count()}")
        lines.append(f"device_name: {torch.cuda.get_device_name(0)}")
        try:
            lines.append(f"cuda_runtime_version: {torch.version.cuda}")
        except Exception:
            pass
    return "\n".join(lines) + "\n"


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def torch_gemm_ref(A: torch.Tensor, B: torch.Tensor, ta: bool, tb: bool) -> torch.Tensor:
    A2 = A.t() if ta else A
    B2 = B.t() if tb else B
    return A2 @ B2


# ----------------------------
# Tensor factories
# ----------------------------
def make_A_B_contig(M: int, K: int, N: int, ta: bool, tb: bool, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda:0")

    if not ta:
        A = torch.randn(M, K, device=device, dtype=dtype).contiguous()
    else:
        A = torch.randn(K, M, device=device, dtype=dtype).contiguous()

    if not tb:
        B = torch.randn(K, N, device=device, dtype=dtype).contiguous()
    else:
        B = torch.randn(N, K, device=device, dtype=dtype).contiguous()

    return A, B


def make_A_B_strided_view(M: int, K: int, N: int, ta: bool, tb: bool, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Non-contiguous *views* to exercise rs/cs indexing (esp. f32 strided path).
    """
    device = torch.device("cuda:0")

    if not ta:
        A_base = torch.randn(M, K * 2, device=device, dtype=dtype).contiguous()
        A = A_base[:, ::2]  # (M,K), Acs=2
    else:
        A_base = torch.randn(K, M * 2, device=device, dtype=dtype).contiguous()
        A = A_base[:, ::2]  # (K,M)

    if not tb:
        B_base = torch.randn(K * 2, N, device=device, dtype=dtype).contiguous()
        B = B_base[::2, :]  # (K,N), Brs doubled
    else:
        B_base = torch.randn(N, K * 2, device=device, dtype=dtype).contiguous()
        B = B_base[:, ::2]  # (N,K)

    return A, B


def make_C(M: int, N: int, dtype: torch.dtype, contig: bool) -> torch.Tensor:
    if contig:
        return torch.empty(M, N, device="cuda:0", dtype=dtype).contiguous()
    # Force a non-contiguous 2D view with same logical shape (M,N)
    base = torch.empty(M, N * 2, device="cuda:0", dtype=dtype).contiguous()
    return base[:, ::2]  # non-contig view (M,N)


# ----------------------------
# Timing helpers
# ----------------------------
@torch.no_grad()
def time_op_call_group(
    op_kind: int,
    inputs: List[torch.Tensor],
    outputs: List[torch.Tensor],
    attr: bytes,
    warmup: int = 50,
    iters: int = 200,
    groups: int = 5,
) -> Dict[str, float]:
    """
    Measure grouped time:
      - warmup
      - repeat groups:
          start, run iters times (no per-iter sync), end, single sync
      - report avg/min/max per-call (ms)
    This is far less noisy than per-iter synchronize.
    """
    # warmup
    for _ in range(warmup):
        _C.op_call(op_kind, inputs, outputs, 0, attr, 0)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    per_call_ms: List[float] = []
    for _g in range(groups):
        starter.record()
        for _ in range(iters):
            _C.op_call(op_kind, inputs, outputs, 0, attr, 0)
        ender.record()
        torch.cuda.synchronize()
        total_ms = float(starter.elapsed_time(ender))
        per_call_ms.append(total_ms / float(iters))

    return {
        "avg_ms": sum(per_call_ms) / len(per_call_ms),
        "min_ms": min(per_call_ms),
        "max_ms": max(per_call_ms),
        "groups": float(groups),
        "iters_per_group": float(iters),
        "warmup": float(warmup),
    }


def gflops(M: int, K: int, N: int, time_ms: float) -> float:
    # GEMM FLOPs ~ 2*M*N*K
    # GFLOPs = flops / (sec * 1e9)
    # sec = ms / 1e3
    flops = 2.0 * float(M) * float(N) * float(K)
    sec = float(time_ms) / 1e3
    if sec <= 0:
        return 0.0
    return flops / (sec * 1e9)


# ----------------------------
# Bench cases
# ----------------------------
def bench_case(
    M: int, K: int, N: int,
    dtype: torch.dtype,
    ta: bool,
    tb: bool,
    *,
    use_strided_ab: bool,
    c_contig: bool,
    do_correctness: bool,
    warmup: int,
    iters: int,
    groups: int,
) -> Dict[str, Any]:
    if use_strided_ab:
        A, B = make_A_B_strided_view(M, K, N, ta, tb, dtype)
    else:
        A, B = make_A_B_contig(M, K, N, ta, tb, dtype)

    C = make_C(M, N, dtype, contig=c_contig)

    delta = None
    if do_correctness:
        if dtype == torch.float16:
            C_ref = torch_gemm_ref(A.float(), B.float(), ta, tb).half()
        else:
            C_ref = torch_gemm_ref(A, B, ta, tb)
        _C.op_call(int(_C.OpKind.Gemm), [A, B], [C], 0, pack_gemm(ta, tb), 0)
        delta = maxabs_delta(C, C_ref)

    timing = time_op_call_group(
        int(_C.OpKind.Gemm),
        [A, B],
        [C],
        pack_gemm(ta, tb),
        warmup=warmup,
        iters=iters,
        groups=groups,
    )

    # meta
    def tmeta(T: torch.Tensor) -> Dict[str, Any]:
        return {
            "shape": list(T.shape),
            "dtype": str(T.dtype).replace("torch.", ""),
            "is_contig": bool(T.is_contiguous()),
            "stride": list(T.stride()),
        }

    avg_ms = float(timing["avg_ms"])
    out = {
        "M": M, "K": K, "N": N,
        "dtype": str(dtype).replace("torch.", ""),
        "ta": int(ta),
        "tb": int(tb),
        "use_strided_ab": bool(use_strided_ab),
        "c_contig": bool(c_contig),
        "A": tmeta(A),
        "B": tmeta(B),
        "C": tmeta(C),
        "delta_maxabs": delta,
        **timing,
        "gflops": gflops(M, K, N, avg_ms),
    }
    return out


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    run = ArtifactRun.create(tag="cuda_binding_gemm_bench_v2")

    meta = {
        "tag": run.tag,
        "root": str(ROOT),
        "script": str(THIS.relative_to(ROOT)),
        "op_kind": int(_C.OpKind.Gemm),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    run.write_json("meta.json", meta)
    run.write_text("env.txt", dump_env())

    warmup = int(os.environ.get("AICF_WARMUP", "50"))
    iters = int(os.environ.get("AICF_ITERS", "200"))
    groups = int(os.environ.get("AICF_GROUPS", "5"))

    # Shape sets: include larger ones to reduce timing noise
    shapes_f32 = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (256, 128, 512),
    ]
    shapes_f16 = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (128, 130, 128),  # K not multiple of 16 -> padding path
    ]

    # correctness sampling only (keep rough)
    correctness_samples = set([
        ("float32", 64, 64, 64, 0, 0, False, True),
        ("float16", 64, 64, 64, 0, 0, False, True),
    ])

    def should_check(dtype: torch.dtype, M: int, K: int, N: int, ta: bool, tb: bool, use_strided_ab: bool, c_contig: bool) -> bool:
        key = (str(dtype).replace("torch.", ""), M, K, N, int(ta), int(tb), bool(use_strided_ab), bool(c_contig))
        return key in correctness_samples

    rows: List[Dict[str, Any]] = []

    # ---- F32 ----
    for (M, K, N) in shapes_f32:
        for (ta, tb) in [(False, False), (True, False)]:
            # contig AB
            rows.append(bench_case(
                M, K, N,
                torch.float32,
                ta, tb,
                use_strided_ab=False,
                c_contig=True,
                do_correctness=should_check(torch.float32, M, K, N, ta, tb, False, True),
                warmup=warmup, iters=iters, groups=groups,
            ))
            # strided AB
            rows.append(bench_case(
                M, K, N,
                torch.float32,
                ta, tb,
                use_strided_ab=True,
                c_contig=True,
                do_correctness=False,
                warmup=warmup, iters=iters, groups=groups,
            ))

    # ---- F16 TC out_f16 ----
    for (M, K, N) in shapes_f16:
        for (ta, tb) in [(False, False), (True, False)]:
            rows.append(bench_case(
                M, K, N,
                torch.float16,
                ta, tb,
                use_strided_ab=False,
                c_contig=True,
                do_correctness=should_check(torch.float16, M, K, N, ta, tb, False, True),
                warmup=warmup, iters=iters, groups=groups,
            ))

    # ---- NEG: f16 with non-contig C should result in "no variant" => NotImplemented ----
    neg: Dict[str, Any]
    try:
        M, K, N = 64, 64, 64
        A, B = make_A_B_contig(M, K, N, False, False, torch.float16)
        C_bad = make_C(M, N, torch.float16, contig=False)
        _C.op_call(int(_C.OpKind.Gemm), [A, B], [C_bad], 0, pack_gemm(0, 0), 0)
        neg = {"neg_f16_c_noncontig": "UNEXPECTED_OK"}
    except RuntimeError as e:
        msg0 = str(e).splitlines()[0] if str(e) else ""
        # We *expect* NotImplemented/no-variant style failure.
        neg = {"neg_f16_c_noncontig": "OK", "error": msg0}

    run.write_json("timing.json", {"rows": rows, "neg": neg})

    # ---- print summary ----
    print("Gemm enum value =", int(_C.OpKind.Gemm))
    print(f"[bench] warmup={warmup} iters/group={iters} groups={groups}")

    def fmt(r: Dict[str, Any]) -> str:
        avg = float(r["avg_ms"])
        return (
            f"{r['dtype']:7s} "
            f"M{r['M']:4d} K{r['K']:4d} N{r['N']:4d} "
            f"ta{r['ta']} tb{r['tb']} "
            f"strAB{int(r['use_strided_ab'])} Cc{int(r['c_contig'])} "
            f"avg{avg:.4f}ms min{float(r['min_ms']):.4f} max{float(r['max_ms']):.4f} "
            f"{float(r['gflops']):8.2f} GF/s"
            + (f" delta{float(r['delta_maxabs']):.3e}" if r["delta_maxabs"] is not None else "")
        )

    for r in rows:
        print(fmt(r))
    print("[neg]", neg)
    print("[artifacts]", str(run.out_dir))


if __name__ == "__main__":
    main()
