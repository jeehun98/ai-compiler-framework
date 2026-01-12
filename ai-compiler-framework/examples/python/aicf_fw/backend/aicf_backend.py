# examples/python/aicf_fw/backend/aicf_backend.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from .base import Backend


def _bootstrap_aicf_cuda():
    repo_root = Path(__file__).resolve().parents[4]
    pymod_dir = repo_root / "build" / "python"
    pkg_dir = pymod_dir / "aicf_cuda"

    if str(pymod_dir) not in sys.path:
        sys.path.insert(0, str(pymod_dir))

    if os.name == "nt":
        os.add_dll_directory(str(pymod_dir))
        os.add_dll_directory(str(pkg_dir))


_bootstrap_aicf_cuda()
import aicf_cuda  # noqa: E402


class AICFBackend(Backend):
    """
    Backend adapter for aicf_cuda._C.op_call

    NOTE:
      - trace is collected in C++ binding (authoritative)
      - python only proxies trace_get/reset/enable
    """

    def __init__(self):
        pass

    # ---- trace helpers (C++ authoritative) ----
    def trace_reset(self) -> None:
        aicf_cuda._C.trace_reset()

    def trace_get(self) -> List[str]:
        return list(aicf_cuda._C.trace_get())

    def trace_enable(self, flag: bool = True) -> None:
        aicf_cuda._C.trace_enable(bool(flag))

    # ----------------------------
    # Core execution path (single-output convenience)
    # ----------------------------
    def op_call(self, op: str, inputs: List[Any], attrs: Dict[str, Any]) -> Any:
        from aicf_fw.core.autograd import in_capture

        if in_capture():
            raise RuntimeError(
                f"AICFBackend.op_call('{op}') is forbidden during capture. "
                "Use op_call_out() with a preallocated output buffer."
            )

        op_l = self._normalize_op(op)
        inputs_t = self._prepare_inputs(op_l, inputs)
        kind, out_desc = self._resolve_op(op_l, inputs_t, attrs)

        if op_l in ("sgd_step", "sgdstep"):
            raise RuntimeError("AICFBackend.op_call: 'sgd_step' must be called via op_call_out (in-place).")

        out = self._allocate_output(out_desc, inputs_t, attrs)

        try:
            aicf_cuda._C.op_call(kind, inputs_t, [out], attrs)
        except Exception as e:
            raise RuntimeError(self._format_fail(op_l, kind, attrs, inputs_t, [out], e)) from e

        return out

    # ----------------------------
    # Core execution path (explicit outputs)
    # ----------------------------
    def op_call_out(self, op: str, inputs: List[Any], outputs: List[Any], attrs: Dict[str, Any]) -> None:
        op_l = self._normalize_op(op)
        warmup = os.getenv("AICF_WARMUP", "0") == "1"

        # warmup safe-guard: do not update params during warmup
        # (compile lowering should already set lr=0, but this doubles safety)
        if warmup and op_l in ("adam_step", "adamstep"):
            attrs = dict(attrs)
            attrs["lr"] = 0.0

        inputs_t = self._prepare_inputs(op_l, inputs)
        outputs_t = [self._as_torch(x) for x in outputs]

        # ✅ warmup safe-guard: do not advance step during warmup (NO-OP)
        # step_inc(in_step) -> out_step (scalar tensor)
        if warmup and op_l in ("step_inc", "stepinc"):
            if len(inputs_t) != 1 or len(outputs_t) != 1:
                raise RuntimeError(
                    f"AICFBackend.op_call_out: step_inc expects 1 input/1 output, got {len(inputs_t)}/{len(outputs_t)}"
                )
            # pointer-stable, capture-safe
            outputs_t[0].copy_(inputs_t[0])
            return

        kind, _ = self._resolve_op(op_l, inputs_t, attrs)

        try:
            aicf_cuda._C.op_call(kind, inputs_t, outputs_t, attrs)
        except Exception as e:
            raise RuntimeError(self._format_fail(op_l, kind, attrs, inputs_t, outputs_t, e)) from e

    # ----------------------------
    # Input preparation (capture-safe)
    # ----------------------------
    def _prepare_inputs(self, op_l: str, inputs: List[Any]) -> List[torch.Tensor]:
        return [self._as_torch(x) for x in inputs]

    # ----------------------------
    # Helpers
    # ----------------------------
    def ones_like(self, x: Any) -> Any:
        x = self._as_torch(x)
        return torch.ones_like(x)

    def _as_torch(self, x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        raise TypeError(f"AICFBackend expects torch.Tensor, got {type(x)}")

    def _normalize_op(self, op: str) -> str:
        return op.strip().lower().replace("-", "_")

    def _format_fail(
        self,
        op: str,
        kind: Any,
        attrs: Dict[str, Any],
        inputs_t: List[torch.Tensor],
        outputs_t: List[torch.Tensor],
        e: Exception,
    ) -> str:
        def _ti(t: torch.Tensor) -> str:
            return (
                f"shape={tuple(t.shape)} stride={tuple(t.stride())} "
                f"dtype={t.dtype} device={t.device} contig={t.is_contiguous()}"
            )

        msg = (
            f"\n[AICFBackend.op_call FAILED]\n"
            f"op={op} kind={kind}\n"
            f"attrs={attrs}\n"
            f"inputs:\n  " + "\n  ".join(_ti(t) for t in inputs_t) + "\n"
            f"outputs:\n  " + "\n  ".join(_ti(t) for t in outputs_t) + "\n"
            f"orig_error: {repr(e)}\n"
        )
        return msg

    # ----------------------------
    # CUDA Graph control + autograd capture guard
    # ----------------------------
    def capture_begin(self):
        from aicf_fw.core.autograd import _set_capture_guard

        _set_capture_guard(True)
        # trace reset happens in C++ capture_begin too, but keep explicit
        self.trace_reset()
        aicf_cuda._C.capture_begin()

    def capture_end(self):
        from aicf_fw.core.autograd import _set_capture_guard

        aicf_cuda._C.capture_end()
        _set_capture_guard(False)

    def replay(self):
        aicf_cuda._C.replay()

    def capture_reset(self):
        from aicf_fw.core.autograd import _set_capture_guard

        aicf_cuda._C.capture_reset()
        _set_capture_guard(False)
        self.trace_reset()

    # ----------------------------
    # Op resolution (UPDATED: copy_saved/copy_aux)
    # ----------------------------
    def _resolve_op(
        self,
        op_l: str,
        inputs: List[torch.Tensor],
        attrs: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        C = aicf_cuda._C

        if op_l in ("add", "eltwiseadd", "eltwise_add"):
            return C.OpKind.EltwiseAdd, {"like": 0}
        if op_l in ("relu", "eltwiserelu", "eltwise_relu"):
            return C.OpKind.EltwiseRelu, {"like": 0}
        if op_l in ("relu_bwd", "relubwd", "relu_backward"):
            return C.OpKind.ReluBwd, {"like": 0}
        if op_l in ("gemm",):
            return C.OpKind.Gemm, {"gemm": True}
        if op_l in ("bias_add", "biasadd"):
            return C.OpKind.BiasAdd, {"like": 0}
        if op_l in ("reduce_sum", "reducesum"):
            return C.OpKind.ReduceSum, {"reduce": True}
        if op_l in ("mse_grad", "msegrad"):
            return C.OpKind.MseGrad, {"like": 0}
        if op_l in ("sgd_step", "sgdstep"):
            return C.OpKind.SgdStep, {"like": 0}

        # UPDATED: copy variants
        if op_l in ("copy", "copy_saved", "copy_aux"):
            return C.OpKind.Copy, {"like": 0}

        if op_l in ("grad_zero", "zero_grad"):
            return C.OpKind.GradZero, {"like": 0}
        if op_l in ("step_inc", "stepinc"):
            return C.OpKind.StepInc, {}
        if op_l in ("bias_corr", "biascorr"):
            return C.OpKind.BiasCorr, {}
        if op_l in ("adam_step", "adamstep"):
            return C.OpKind.AdamStep, {"like": 0}

        raise KeyError(f"AICFBackend: unknown op '{op_l}'")

    def _allocate_output(
        self,
        desc: Dict[str, Any],
        inputs: List[torch.Tensor],
        attrs: Dict[str, Any],
    ) -> torch.Tensor:
        if "like" in desc:
            ref = inputs[desc["like"]]
            return torch.empty_like(ref)

        if "gemm" in desc:
            A, B = inputs
            transA = bool(attrs.get("transA", False))
            transB = bool(attrs.get("transB", False))

            if A.dim() != 2 or B.dim() != 2:
                raise RuntimeError(f"GEMM expects 2D tensors, got A.dim={A.dim()} B.dim={B.dim()}")

            M = A.shape[1] if transA else A.shape[0]
            K_a = A.shape[0] if transA else A.shape[1]
            K_b = B.shape[1] if transB else B.shape[0]
            N = B.shape[0] if transB else B.shape[1]

            if K_a != K_b:
                raise RuntimeError(
                    f"GEMM mismatch: K_a={K_a} K_b={K_b} "
                    f"A.shape={tuple(A.shape)} B.shape={tuple(B.shape)} attrs={attrs}"
                )

            return torch.empty((M, N), device=A.device, dtype=A.dtype)

        if "reduce" in desc:
            X = inputs[0]
            axis = int(attrs["axis"])
            keepdim = bool(attrs.get("keepdim", False))
            if axis < 0:
                axis += X.dim()

            out_shape = list(X.shape)
            if keepdim:
                out_shape[axis] = 1
            else:
                out_shape.pop(axis)

            return torch.empty(out_shape, device=X.device, dtype=X.dtype)

        raise RuntimeError(f"Cannot allocate output for desc={desc}")
