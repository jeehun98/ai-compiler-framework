from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from aicf_cuda import _C

from .tensor import Tensor
from .torch_backend import BackendConfig  # config 공유
from .utils.profiling import OpProfiler


class AicfBackend:
    """
    AICF backend (v0.2)
    - op_call 기반 실행
    - Trainer(mode="capture")와 1:1로 맞물리도록:
        * prepare_capture_batch(batch): 고정 입력 버퍼 생성/반환
        * bind_batch(batch): 매 step 입력 copy_로 갱신
        * capture_begin/end/replay: _C.capture_* 사용
    - 주의:
        * capture 구간 내에서 새로운 tensor 할당/contiguous 변환 금지
        * 따라서 준비 단계(prepare_capture_batch)에서 필요한 contig/shape 고정
    """

    def __init__(self, cfg: Optional[BackendConfig] = None) -> None:
        self.cfg = cfg or BackendConfig()
        self.profiler = OpProfiler(enabled=self.cfg.enable_profiler)

        # capture fixed input buffers (torch.Tensor)
        self._cap_x: Optional[torch.Tensor] = None
        self._cap_t: Optional[torch.Tensor] = None

        # debug/stats
        self._captured: bool = False

    # -----------------
    # mode
    # -----------------
    def set_mode(self, mode: str) -> None:
        self.cfg.mode = mode

    # -----------------
    # capture contracts (Trainer expects these)
    # -----------------
    def prepare_capture_batch(self, batch) -> Tuple[Tensor, Tensor]:
        """
        Create fixed input buffers for CUDA Graph capture.
        - 반드시 이 텐서 포인터/shape가 캡처 동안 고정되어야 함.
        - 이후 매 step bind_batch에서 copy_만 수행.
        """
        x, t = batch
        x_t = x.t if hasattr(x, "t") else x
        t_t = t.t if hasattr(t, "t") else t

        if not isinstance(x_t, torch.Tensor) or not isinstance(t_t, torch.Tensor):
            raise TypeError("prepare_capture_batch expects Tensor wrapper or torch.Tensor")

        # clone to own buffers (fixed pointers), enforce contiguous here (allowed: outside capture)
        self._cap_x = x_t.clone().contiguous()
        self._cap_t = t_t.clone().contiguous()

        return Tensor(self._cap_x), Tensor(self._cap_t)

    def bind_batch(self, batch) -> None:
        """
        Update fixed buffers by copy_. This must be capture-safe for replay phase.
        """
        if self._cap_x is None or self._cap_t is None:
            raise RuntimeError("bind_batch called before prepare_capture_batch")

        x, t = batch
        x_t = x.t if hasattr(x, "t") else x
        t_t = t.t if hasattr(t, "t") else t

        # IMPORTANT: copy_ requires same shape/dtype/device
        if x_t.shape != self._cap_x.shape or x_t.dtype != self._cap_x.dtype or x_t.device != self._cap_x.device:
            raise RuntimeError(
                f"bind_batch: x mismatch. expected {tuple(self._cap_x.shape)}/{self._cap_x.dtype}/{self._cap_x.device}, "
                f"got {tuple(x_t.shape)}/{x_t.dtype}/{x_t.device}"
            )
        if t_t.shape != self._cap_t.shape or t_t.dtype != self._cap_t.dtype or t_t.device != self._cap_t.device:
            raise RuntimeError(
                f"bind_batch: t mismatch. expected {tuple(self._cap_t.shape)}/{self._cap_t.dtype}/{self._cap_t.device}, "
                f"got {tuple(t_t.shape)}/{t_t.dtype}/{t_t.device}"
            )

        # copy only (no allocations)
        self._cap_x.copy_(x_t)
        self._cap_t.copy_(t_t)

    # optional alias for Trainer compatibility shim
    def set_inputs(self, batch) -> None:
        self.bind_batch(batch)

    def warmup(self, model: Any, sample_batch: Any) -> None:
        # keep it minimal; allocator warmup should be done outside capture if needed.
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
    # op_call helpers
    # -----------------
    def _call(self, kind: _C.OpKind, inputs, outputs, attrs: Optional[Dict[str, Any]] = None) -> None:
        _C.op_call(kind, inputs, outputs, attrs or {})

    def _sig(self, op_name: str, *tensors, **kwargs) -> str:
        parts = [op_name] + [tt.signature() for tt in tensors]
        kv = {k: v for k, v in kwargs.items() if v is not None and k != "attrs"}
        if kv:
            parts.append(str(kv))
        return "|".join(parts)

    def _require_contig(self, *ts: torch.Tensor, what: str) -> None:
        # capture-safe contract: do not call contiguous() here.
        for t in ts:
            if not t.is_contiguous():
                raise RuntimeError(f"{what}: requires contiguous tensors (got non-contiguous)")

    # -----------------
    # ops (minimal set for your current path)
    # -----------------
    def relu(self, x: Tensor) -> Tensor:
        sig = self._sig("relu", x)
        with self.profiler.scope("relu", sig, mode=self.cfg.mode):
            X = x.t
            if not X.is_contiguous():
                X = X.contiguous()

            orig_shape = X.shape
            X1 = X.view(-1).contiguous()          # ✅ 1D
            Y1 = torch.empty_like(X1).contiguous()

            self._call(_C.OpKind.EltwiseRelu, [X1], [Y1], {})

            Y = Y1.view(orig_shape)              # ✅ 원 shape 복구
            return Tensor(Y)


    def gemm(
        self,
        a: Tensor,
        b: Tensor,
        bias: Optional[Tensor] = None,
        act: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        TorchBackend contract:
          y = a.t @ b.t
          + optional bias
          + optional relu
        """
        sig = self._sig("gemm", a, b, bias=bias, act=act, attrs=attrs)
        with self.profiler.scope("gemm", sig, mode=self.cfg.mode):
            A = a.t
            B = b.t
            self._require_contig(A, B, what="gemm")

            # output allocation:
            # - For capture correctness, allocations should be avoided inside capture region.
            #   In practice, capture path should use fixed buffers; but Trainer currently captures train_step_eager()
            #   which will allocate unless you preallocate in model/ops. v0.2에서는 우선 eager/bench 우선.
            C = torch.empty((A.shape[0], B.shape[1]), device=A.device, dtype=A.dtype)

            self._call(_C.OpKind.Gemm, [A, B], [C], attrs or {})

            if bias is not None:
                bias_t = bias.t
                self._require_contig(bias_t, what="bias_add")
                C2 = torch.empty_like(C)
                # 일반적인 linear bias: [N] broadcast over last dim
                self._call(_C.OpKind.BiasAdd, [C, bias_t], [C2], {"axis": 1})
                C = C2

            if act == "relu":
                C3 = torch.empty_like(C)
                self._call(_C.OpKind.EltwiseRelu, [C], [C3], {})
                C = C3

            return Tensor(C)

    def mse(self, y: Tensor, t: Tensor) -> Tensor:
        """
        현재 커널셋에 mse forward가 없으면 torch fallback 유지.
        (mse_grad + reduce_sum으로 forward도 구성 가능하지만, 지금 단계에선 간단히 둔다)
        """
        sig = self._sig("mse", y, t)
        with self.profiler.scope("mse", sig, mode=self.cfg.mode):
            return Tensor(torch.mean((y.t - t.t) ** 2))
