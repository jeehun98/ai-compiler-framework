from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from aicf_cuda import _C

from .tensor import Tensor
from .torch_backend import BackendConfig
from .utils.profiling import OpProfiler


class AicfBackend:
    def __init__(self, cfg: Optional[BackendConfig] = None) -> None:
        self.cfg = cfg or BackendConfig()
        self.profiler = OpProfiler(enabled=self.cfg.enable_profiler)

        self._cap_x: Optional[torch.Tensor] = None
        self._cap_t: Optional[torch.Tensor] = None
        self._captured: bool = False

    def set_mode(self, mode: str) -> None:
        self.cfg.mode = mode

    # -----------------
    # capture
    # -----------------
    def prepare_capture_batch(self, batch) -> Tuple[Tensor, Tensor]:
        x, t = batch
        x_t = x.t if hasattr(x, "t") else x
        t_t = t.t if hasattr(t, "t") else t

        if not isinstance(x_t, torch.Tensor) or not isinstance(t_t, torch.Tensor):
            raise TypeError("prepare_capture_batch expects Tensor wrapper or torch.Tensor")

        # capture-safe: allocate once, keep contiguous buffers
        self._cap_x = x_t.clone().contiguous()
        self._cap_t = t_t.clone().contiguous()
        return Tensor(self._cap_x), Tensor(self._cap_t)

    def bind_batch(self, batch) -> None:
        if self._cap_x is None or self._cap_t is None:
            raise RuntimeError("bind_batch called before prepare_capture_batch")

        x, t = batch
        x_t = x.t if hasattr(x, "t") else x
        t_t = t.t if hasattr(t, "t") else t

        if x_t.shape != self._cap_x.shape or x_t.dtype != self._cap_x.dtype or x_t.device != self._cap_x.device:
            raise RuntimeError("bind_batch: x mismatch")
        if t_t.shape != self._cap_t.shape or t_t.dtype != self._cap_t.dtype or t_t.device != self._cap_t.device:
            raise RuntimeError("bind_batch: t mismatch")

        self._cap_x.copy_(x_t)
        self._cap_t.copy_(t_t)

    def set_inputs(self, batch) -> None:
        self.bind_batch(batch)

    def warmup(self, model: Any, sample_batch: Any) -> None:
        return

    def capture_begin(self) -> None:
        self.set_mode("capture")
        _C.capture_begin()
        self._captured = False

    def capture_end(self) -> None:
        _C.capture_end()
        self._captured = True

    def replay(self) -> None:
        if not self._captured:
            raise RuntimeError("replay called but graph not captured yet")
        _C.replay()

    # -----------------
    # low-level call
    # -----------------
    def _call(self, kind: _C.OpKind, inputs, outputs, attrs=None) -> None:
        try:
            _C.op_call(kind, inputs, outputs, attrs or {})
        except Exception as e:
            def _desc(t: torch.Tensor) -> str:
                return f"shape={tuple(t.shape)} dtype={t.dtype} contig={t.is_contiguous()} stride={tuple(t.stride())}"
            in_s = ", ".join([_desc(t) for t in inputs])
            out_s = ", ".join([_desc(t) for t in outputs])
            raise RuntimeError(
                f"op_call FAILED kind={int(kind)} attrs={attrs or {}}\n"
                f"  inputs:  {in_s}\n"
                f"  outputs: {out_s}\n"
                f"  orig: {e}"
            )

    def _require_contig(self, *ts: torch.Tensor, what: str) -> None:
        # capture-safe contract: do not call contiguous() here.
        for t in ts:
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"{what}: expected torch.Tensor, got {type(t)}")
            if not t.is_contiguous():
                raise RuntimeError(f"{what}: requires contiguous tensors (got non-contiguous)")

    def _sig(self, op_name: str, *tensors, **kwargs) -> str:
        parts = [op_name] + [tt.signature() for tt in tensors]
        kv = {k: v for k, v in kwargs.items() if v is not None and k != "attrs"}
        if kv:
            parts.append(str(kv))
        return "|".join(parts)

    # -----------------
    # helpers (capture-safe rules)
    # -----------------
    def _is_capture_like(self) -> bool:
        # treat capture & replay as "no allocation / no dtype transform / no contiguous()"
        return self.cfg.mode in ("capture", "replay")

    def _forbid_transform_in_capture(self, op: str, why: str) -> None:
        if self._is_capture_like():
            raise RuntimeError(f"{op}: forbidden in {self.cfg.mode} mode: {why}")

    # -----------------
    # ops
    # -----------------
    def relu(self, x: Tensor) -> Tensor:
        sig = self._sig("relu", x)
        with self.profiler.scope("relu", sig, mode=self.cfg.mode):
            X = x.t
            self._require_contig(X, what="relu")
            Y = torch.empty_like(X)
            self._call(_C.OpKind.EltwiseRelu, [X], [Y], {})
            return Tensor(Y)

    def gemm(
        self,
        a: Tensor,
        b: Tensor,
        bias: Optional[Tensor] = None,
        act: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        sig = self._sig("gemm", a, b, bias=bias, act=act, attrs=attrs)
        with self.profiler.scope("gemm", sig, mode=self.cfg.mode):
            A = a.t
            B = b.t
            self._require_contig(A, B, what="gemm")

            attrs = dict(attrs or {})
            transA = bool(attrs.get("transA", False))
            transB = bool(attrs.get("transB", False))

            # --- dtype policy ---
            # If any input is f16 -> use TC path (f16 inputs) and keep output dtype = f16
            # (because you added TC out_f16 variants and want to keep f16 chain)
            want_tc = (A.dtype == torch.float16) or (B.dtype == torch.float16)

            # capture-safe: do not .to() or .contiguous() inside
            if want_tc:
                if A.dtype != torch.float16:
                    self._forbid_transform_in_capture("gemm", "A.dtype cast to f16 required")
                    A = A.to(torch.float16).contiguous()
                if B.dtype != torch.float16:
                    self._forbid_transform_in_capture("gemm", "B.dtype cast to f16 required")
                    B = B.to(torch.float16).contiguous()

                # --- transB storage contract enforcement ---
                # Your TC NT(out_f16) path expects:
                #   transB=True -> B is stored as [N,K] contiguous
                # But callers sometimes pass [K,N] contiguous.
                # Fix it only in eager mode (capture-safe prohibits this).
                if transB:
                    # We need K from A shape (if not transA), otherwise from A storage shape.
                    if transA:
                        # A storage is [K,M], so K = A.shape[0]
                        K = A.shape[0]
                    else:
                        # A is [M,K]
                        K = A.shape[1]

                    # If B looks like [K,N] (i.e., B.shape[0] == K), transpose storage.
                    if B.dim() == 2 and B.shape[0] == K:
                        self._forbid_transform_in_capture("gemm", "transB=True requires B_storage=[N,K]; need B.t().contiguous()")
                        B = B.t().contiguous()  # now [N,K]

                # infer output shape
                M = (A.shape[1] if transA else A.shape[0])  # logical M
                if transB:
                    # B storage [N,K] => logical N = B.shape[0]
                    N = B.shape[0]
                else:
                    # B [K,N]
                    N = B.shape[1]

                # TC out_f16 variants expect output f16
                C = torch.empty((M, N), device=A.device, dtype=torch.float16).contiguous()
                self._call(_C.OpKind.Gemm, [A, B], [C], attrs)

            else:
                # f32 fallback path
                if A.dtype != torch.float32:
                    self._forbid_transform_in_capture("gemm", "A.dtype cast to f32 required")
                    A = A.to(torch.float32).contiguous()
                if B.dtype != torch.float32:
                    self._forbid_transform_in_capture("gemm", "B.dtype cast to f32 required")
                    B = B.to(torch.float32).contiguous()

                if transB:
                    # fallback expects transB kernel handles stride OR storage; your naive transB uses strides.
                    # but current gemm_f32_variant_check requires B shape [N,K] with strides set.
                    # Here we assume B is already storage [N,K] if transB=True.
                    N = B.shape[0]
                    M = (A.shape[1] if transA else A.shape[0])
                else:
                    N = B.shape[1]
                    M = (A.shape[1] if transA else A.shape[0])

                C = torch.empty((M, N), device=A.device, dtype=torch.float32).contiguous()
                self._call(_C.OpKind.Gemm, [A, B], [C], attrs)

            # ---- BiasAdd ----
            if bias is not None:
                bias_t = bias.t
                self._require_contig(bias_t, what="bias_add")
                if bias_t.dtype != C.dtype:
                    self._forbid_transform_in_capture("bias_add", "bias dtype cast required")
                    bias_t = bias_t.to(C.dtype).contiguous()
                C2 = torch.empty_like(C)
                self._call(_C.OpKind.BiasAdd, [C, bias_t], [C2], {"axis": -1})
                C = C2

            # ---- ReLU ----
            if act == "relu":
                C3 = torch.empty_like(C)
                self._call(_C.OpKind.EltwiseRelu, [C], [C3], {})
                C = C3

            return Tensor(C)

    def mse(self, y: Tensor, t: Tensor) -> Tensor:
        sig = self._sig("mse", y, t)
        with self.profiler.scope("mse", sig, mode=self.cfg.mode):
            return Tensor(torch.mean((y.t - t.t) ** 2))
