# examples/python/aicf_fw/core_v2/exec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import os
import sys
from pathlib import Path

import torch

from .ir import IRGraph
from .plan import BindingPlan, allocate_static_env


# -------------------------
# bootstrap: make aicf_cuda importable (build/python)
# -------------------------

def _bootstrap_aicf_cuda():
    # exec.py: .../examples/python/aicf_fw/core_v2/exec.py
    # repo_root: .../ai-compiler-framework
    repo_root = Path(__file__).resolve().parents[4]
    pymod_dir = repo_root / "build" / "python"
    pkg_dir = pymod_dir / "aicf_cuda"

    if pymod_dir.exists() and str(pymod_dir) not in sys.path:
        sys.path.insert(0, str(pymod_dir))

    if os.name == "nt":
        if pymod_dir.exists():
            os.add_dll_directory(str(pymod_dir))
        if pkg_dir.exists():
            os.add_dll_directory(str(pkg_dir))


_bootstrap_aicf_cuda()

from aicf_cuda import _C  # noqa: E402


# -------------------------
# Options / helpers
# -------------------------

@dataclass
class ExecOptions:
    debug: bool = False
    debug_limit: int = 50
    check_shapes: bool = True
    check_device: bool = True
    check_dtype: bool = True


def _tmeta(t: torch.Tensor) -> str:
    return f"ptr={t.data_ptr()} shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}"


def _unwrap(x) -> torch.Tensor:
    return x.data if hasattr(x, "data") else x


def _assert_tensor_matches(spec, t: torch.Tensor, opts: ExecOptions):
    if opts.check_shapes and tuple(t.shape) != tuple(spec.shape):
        raise RuntimeError(
            f"[plan] shape mismatch for vid{spec.vid}({spec.name}): {tuple(t.shape)} != {tuple(spec.shape)}"
        )

    if opts.check_dtype:
        ds = str(t.dtype)
        if ("float16" in spec.dtype and "float16" not in ds) or \
           ("bfloat16" in spec.dtype and "bfloat16" not in ds) or \
           ("float32" in spec.dtype and "float32" not in ds) or \
           ("float64" in spec.dtype and "float64" not in ds) or \
           ("int32" in spec.dtype and "int32" not in ds) or \
           ("int64" in spec.dtype and "int64" not in ds):
            raise RuntimeError(
                f"[plan] dtype mismatch for vid{spec.vid}({spec.name}): {t.dtype} vs {spec.dtype}"
            )

    if opts.check_device:
        if "cuda" in spec.device and (not t.is_cuda):
            raise RuntimeError(
                f"[plan] device mismatch for vid{spec.vid}({spec.name}): expected cuda, got {t.device}"
            )


def _canon_name(s: str) -> str:
    s = str(s)
    if s.startswith("d_"):
        s = s[2:]
    return s


def _build_name_maps(plan: BindingPlan) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    name_to_vids: Dict[str, List[int]] = {}
    canon_to_vids: Dict[str, List[int]] = {}

    for vid, spec in plan.specs.items():
        n = str(spec.name)
        name_to_vids.setdefault(n, []).append(int(vid))

        cn = _canon_name(n)
        canon_to_vids.setdefault(cn, []).append(int(vid))

    return name_to_vids, canon_to_vids


def _collect_vid_candidates_for_key(
    *,
    key: str,
    name_to_vids: Dict[str, List[int]],
    canon_to_vids: Dict[str, List[int]],
) -> List[int]:
    k = str(key)

    vids = name_to_vids.get(k)
    if vids:
        return list(vids)

    ck = _canon_name(k)
    vids = canon_to_vids.get(ck)
    if vids:
        return list(vids)

    out: List[int] = []
    for pn, vs in name_to_vids.items():
        if str(pn).endswith(k):
            out.extend(vs)
    if out:
        return out

    out2: List[int] = []
    for pn, vs in name_to_vids.items():
        if _canon_name(pn).endswith(ck):
            out2.extend(vs)
    return out2


def _spec_kind_matches(spec, kind: str) -> bool:
    role = str(getattr(spec, "role", "") or "")
    if not role:
        return True
    if kind == "input":
        return ("input" in role) or (role == "in")
    if kind == "param":
        return ("param" in role) or ("weight" in role) or ("bias" in role)
    return True


def _choose_vid_for_tensor(
    *,
    candidates: List[int],
    plan: BindingPlan,
    t: torch.Tensor,
    kind: str,
    opts: ExecOptions,
) -> Optional[int]:
    if not candidates:
        return None

    filtered = [v for v in candidates if _spec_kind_matches(plan.specs[int(v)], kind)]
    if not filtered:
        filtered = list(candidates)

    for v in filtered:
        spec = plan.specs[int(v)]
        try:
            _assert_tensor_matches(spec, t, opts)
            return int(v)
        except Exception:
            pass

    return None


# -------------------------
# OpKind mapping (string op -> _C.OpKind int)
# -------------------------

_OPNAME_TO_KIND: Dict[str, int] = {
    "add": int(_C.OpKind.EltwiseAdd),
    "relu": int(_C.OpKind.EltwiseRelu),
    "gemm": int(_C.OpKind.Gemm),
    "bias_add": int(_C.OpKind.BiasAdd),
    "reduce_sum": int(_C.OpKind.ReduceSum),
    "mse_grad": int(_C.OpKind.MseGrad),
    "relu_bwd": int(_C.OpKind.ReluBwd),
    "sgd_step": int(_C.OpKind.SgdStep),
    "copy": int(_C.OpKind.Copy),
    "copy_saved": int(_C.OpKind.Copy),  # ✅ 추가
    "copy_aux": int(_C.OpKind.Copy),    # ✅ 추가
    "grad_zero": int(_C.OpKind.GradZero),
    "adam_step": int(_C.OpKind.AdamStep),
    "step_inc": int(_C.OpKind.StepInc),
    "bias_corr": int(_C.OpKind.BiasCorr),
    "layernorm_fwd": int(_C.OpKind.LayerNormFwd),
    "layernorm_bwd": int(_C.OpKind.LayerNormBwd),
    "batchnorm_fwd": int(_C.OpKind.BatchNormFwd),
    "batchnorm_bwd": int(_C.OpKind.BatchNormBwd),
}


def _opkind_from_name(op: str) -> int:
    k = str(op).strip().lower()
    if k not in _OPNAME_TO_KIND:
        raise RuntimeError(f"[exec] unsupported lowered op '{op}' (no OpKind mapping)")
    return _OPNAME_TO_KIND[k]


# -------------------------
# Executor
# -------------------------

class PlannedExecutor:
    """
    plan 기반 실행기 (direct aicf_cuda._C)
      - lowered를 _C.op_call(kind, ins, outs, attrs)로 실행
      - graph capture/replay는 _C.graph_* 기반
    """

    def __init__(
        self,
        *,
        ir: IRGraph,
        lowered: List[Dict[str, Any]],
        plan: BindingPlan,
        device: Optional[torch.device] = None,
        opts: Optional[ExecOptions] = None,
    ):
        self.ir = ir
        self.lowered = lowered
        self.plan = plan
        self.device = device
        self.opts = opts or ExecOptions()

        self.env: Dict[int, torch.Tensor] = {}
        self._captured: bool = False

    # ---- trace passthrough (optional) ----
    def trace_enable(self, flag: bool = True) -> None:
        fn = getattr(_C, "trace_enable", None)
        if fn is not None:
            fn(bool(flag))

    def trace_reset(self) -> None:
        fn = getattr(_C, "trace_reset", None)
        if fn is not None:
            fn()

    def trace_get(self) -> List[str]:
        fn = getattr(_C, "trace_get", None)
        if fn is None:
            return []
        return list(fn())

    # ---- env binding ----
    def build_env(
        self,
        *,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        reuse_static: bool = False,
    ) -> Dict[int, torch.Tensor]:
        opts = self.opts

        if (not reuse_static) or (not self.env):
            self.env = allocate_static_env(self.ir, self.plan, device=self.device)

        name_to_vids, canon_to_vids = _build_name_maps(self.plan)

        def bind(src: Dict[str, Any], kind: str):
            for name, obj in src.items():
                t = _unwrap(obj)
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"[plan] {kind} '{name}' must be torch.Tensor-like, got {type(obj)}")

                cands = _collect_vid_candidates_for_key(
                    key=str(name),
                    name_to_vids=name_to_vids,
                    canon_to_vids=canon_to_vids,
                )
                vid = _choose_vid_for_tensor(
                    candidates=cands,
                    plan=self.plan,
                    t=t,
                    kind=kind,
                    opts=opts,
                )
                if vid is None:
                    known = list(name_to_vids.keys())
                    head = known[:24]
                    tail_note = "" if len(known) <= 24 else f" ...(+{len(known)-24})"
                    raise KeyError(
                        f"[plan] unknown or ambiguous {kind} name='{name}'. "
                        f"candidates={cands}. known={head}{tail_note}"
                    )

                spec = self.plan.specs[int(vid)]
                _assert_tensor_matches(spec, t, opts)
                self.env[int(vid)] = t

        bind(inputs, "input")
        bind(params, "param")

        for vid in self.plan.inputs + self.plan.params + self.plan.statics:
            if int(vid) not in self.env:
                spec = self.plan.specs[int(vid)]
                raise RuntimeError(f"[plan] env missing vid{vid} ({spec.name}) role={spec.role}")

        return self.env

    # ---- run ----
    @torch.no_grad()
    def run(
        self,
        *,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        reuse_static: bool = True,
    ) -> Dict[int, torch.Tensor]:
        self.build_env(inputs=inputs, params=params, reuse_static=reuse_static)

        dbg = self.opts.debug
        limit = int(self.opts.debug_limit)

        warmup = os.getenv("AICF_WARMUP", "0") == "1"

        for i, it in enumerate(self.lowered):
            op = str(it["op"])
            kind = _opkind_from_name(op)

            in_vids = [int(x) for x in it.get("inputs", [])]
            out_vids = [int(y) for y in it.get("outputs", [])]
            attrs = dict(it.get("attrs", {}) or {})

            ins_t = [self.env[v] for v in in_vids]
            outs_t = [self.env[v] for v in out_vids]

            if dbg and i < limit:
                ins_s = ", ".join([f"v{v:03d}:{_tmeta(self.env[v])}" for v in in_vids])
                outs_s = ", ".join([f"v{v:03d}:{_tmeta(self.env[v])}" for v in out_vids])
                print(f"[exec][#{i:03d}] {op} kind={kind} attrs={attrs}")
                print(f"  in : {ins_s}")
                print(f"  out: {outs_s}")

            # ✅ warmup (for correctness comparisons):
            # - do not mutate step
            # - do not update params during capture/replay tests
            if warmup and op == "step_inc":
                if len(ins_t) != 1 or len(outs_t) != 1:
                    raise RuntimeError(f"[exec] step_inc expects 1 in / 1 out, got {len(ins_t)} / {len(outs_t)}")
                outs_t[0].copy_(ins_t[0])
                continue

            if warmup and op == "adam_step":
                attrs = dict(attrs)
                attrs["lr"] = 0.0

            _C.op_call(int(kind), ins_t, outs_t, attrs)

            for v, t in zip(out_vids, outs_t):
                self.env[int(v)] = t

        return self.env

    # ---- CUDA Graph ----
    @torch.no_grad()
    def capture(
        self,
        *,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        reuse_static: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Capture lowered execution into CUDA Graph.
        Uses _C.graph_* API (dedicated stream).
        """
        self.build_env(inputs=inputs, params=params, reuse_static=reuse_static)

        s = int(_C.graph_begin())  # returns dedicated cudaStream_t handle (int)
        try:
            # run on dedicated stream while capture is active
            self._run_with_stream(stream=s, reuse_static=True)
        finally:
            _C.graph_end()

        self._captured = True
        return self.env

    @torch.no_grad()
    def _run_with_stream(self, *, stream: int, reuse_static: bool) -> None:
        dbg = self.opts.debug
        limit = int(self.opts.debug_limit)
        warmup = os.getenv("AICF_WARMUP", "0") == "1"

        for i, it in enumerate(self.lowered):
            op = str(it["op"])
            kind = _opkind_from_name(op)

            in_vids = [int(x) for x in it.get("inputs", [])]
            out_vids = [int(y) for y in it.get("outputs", [])]
            attrs = dict(it.get("attrs", {}) or {})

            ins_t = [self.env[v] for v in in_vids]
            outs_t = [self.env[v] for v in out_vids]

            if dbg and i < limit:
                ins_s = ", ".join([f"v{v:03d}:{_tmeta(self.env[v])}" for v in in_vids])
                outs_s = ", ".join([f"v{v:03d}:{_tmeta(self.env[v])}" for v in out_vids])
                print(f"[exec][#{i:03d}] {op} kind={kind} attrs={attrs}")
                print(f"  in : {ins_s}")
                print(f"  out: {outs_s}")

            if warmup and op == "step_inc":
                outs_t[0].copy_(ins_t[0])
                continue

            if warmup and op == "adam_step":
                attrs = dict(attrs)
                attrs["lr"] = 0.0

            _C.op_call(int(kind), ins_t, outs_t, attrs, int(stream))

            for v, t in zip(out_vids, outs_t):
                self.env[int(v)] = t

    @torch.no_grad()
    def replay(self, n: int = 1) -> None:
        if n <= 0:
            return
        if not self._captured:
            raise RuntimeError("[exec] replay called before capture()")
        for _ in range(int(n)):
            _C.graph_launch()

    def reset_graph(self) -> None:
        _C.graph_reset()
        self._captured = False
