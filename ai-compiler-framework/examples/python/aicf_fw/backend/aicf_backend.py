# aicf_fw/backend/aicf_backend.py
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

    - inputs/outputs: torch.Tensor (CUDA)
    - attrs: dict[str, bool|int|float]
    """

    def __init__(self):
        pass

    # ----------------------------
    # Core execution path (single-output convenience)
    # ----------------------------
    def op_call(self, op: str, inputs: List[Any], attrs: Dict[str, Any]) -> Any:
        op_l = self._normalize_op(op)

        # IMPORTANT POLICY:
        # - GEMM must preserve strides/views to support transA/transB without materializing.
        # - Most other ops currently assume contiguous tensors (kernel implementations).
        if op_l == "gemm":
            inputs_t = [self._as_torch(x) for x in inputs]  # keep views/strides
        else:
            inputs_t = [self._as_torch(x).contiguous() for x in inputs]

        kind, out_desc = self._resolve_op(op_l, inputs_t, attrs)

        # sgd_step is in-place only
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

        if op_l == "gemm":
            inputs_t = [self._as_torch(x) for x in inputs]  # keep views/strides
        else:
            inputs_t = [self._as_torch(x).contiguous() for x in inputs]

        # DO NOT force outputs to contiguous:
        # - If user passes a view/output buffer intentionally, .contiguous() would silently
        #   write into a temp and discard results.
        outputs_t = [self._as_torch(x) for x in outputs]

        kind, _ = self._resolve_op(op_l, inputs_t, attrs)

        try:
            aicf_cuda._C.op_call(kind, inputs_t, outputs_t, attrs)
        except Exception as e:
            raise RuntimeError(self._format_fail(op_l, kind, attrs, inputs_t, outputs_t, e)) from e

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
            # include stride for debugging view/transpose issues
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
    # CUDA Graph control
    # ----------------------------
    def capture_begin(self):
        aicf_cuda._C.capture_begin()

    def capture_end(self):
        aicf_cuda._C.capture_end()

    def replay(self):
        aicf_cuda._C.replay()

    def capture_reset(self):
        aicf_cuda._C.capture_reset()

    # ----------------------------
    # Op resolution
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
            return C.OpKind.SgdStep, {"like": 0}  # in-place, like-only

        raise KeyError(f"AICFBackend: unknown op '{op_l}'")

    # ----------------------------
    # Output allocation
    # ----------------------------
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

            # Logical shapes:
            # A_op: (M,K)
            # B_op: (K,N)
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

            # Output dtype follows A (typical). If later you want f16->f16 with f32-acc,
            # keep output f16 (done in CUDA kernel).
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

            # NOTE:
            # If your ReduceSum kernel is f16->f32 for some axes, you can enforce dtype here.
            # For now, keep dtype same as input.
            return torch.empty(out_shape, device=X.device, dtype=X.dtype)

        raise RuntimeError(f"Cannot allocate output for desc={desc}")
