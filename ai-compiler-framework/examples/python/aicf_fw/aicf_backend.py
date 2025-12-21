from __future__ import annotations
from typing import Optional, Any, Dict, Tuple
import os
import torch

from .torch_backend import TorchBackend, BackendConfig
from .tensor import Tensor

try:
    from aicf_cuda import _C as C  # ✅ 너 바인딩: PYBIND11_MODULE(_C, ...)
except Exception:
    C = None


def _unwrap(x: Any) -> torch.Tensor:
    return x.t if hasattr(x, "t") else x


def _wrap(t: torch.Tensor) -> Tensor:
    return t if isinstance(t, Tensor) else Tensor(t)


def _is_cuda_contig_f32(t: torch.Tensor) -> bool:
    return t.is_cuda and t.dtype == torch.float32 and t.is_contiguous()


class AicfBackend:
    """
    - unified op_call backend
    - fallback to TorchBackend when constraints not met
    - capture/replay via _C.capture_* API
    - input binding for replay uses fixed buffers + copy_
    """

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.fallback = TorchBackend(cfg)

        self.enabled = (os.environ.get("AICF_ENABLE_AICF_KERNELS", "1") == "1")
        self.debug = (os.environ.get("AICF_BACKEND_DEBUG", "0") == "1")
        self.mode: str = "eager"

        # capture state
        self._captured: bool = False
        self._capturing: bool = False
        self._x_buf: Optional[torch.Tensor] = None
        self._t_buf: Optional[torch.Tensor] = None

        if C is None:
            if self.debug:
                print("[AICF] aicf_cuda._C not found -> fallback to torch")
            self.enabled = False

    def _log(self, msg: str):
        if self.debug:
            print(msg)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if hasattr(self.fallback, "set_mode"):
            self.fallback.set_mode(mode)

    # -------------------------
    # Capture helpers (forward-only bench / later full-step)
    # -------------------------
    def prepare_capture_batch(self, batch) -> Tuple[Any, Any]:
        """
        CUDA Graph에서 input pointer가 고정돼야 하므로,
        캡처/리플레이는 이 버퍼를 입력으로 사용해야 함.
        """
        x, t = batch
        xt = _unwrap(x)
        tt = _unwrap(t)

        if not (isinstance(xt, torch.Tensor) and isinstance(tt, torch.Tensor)):
            raise RuntimeError("prepare_capture_batch expects torch.Tensor or Tensor wrapper with .t")
        if not xt.is_cuda or not tt.is_cuda:
            raise RuntimeError("capture requires CUDA tensors")

        need_new = (
            self._x_buf is None or self._t_buf is None
            or self._x_buf.shape != xt.shape or self._x_buf.dtype != xt.dtype or self._x_buf.device != xt.device
            or self._t_buf.shape != tt.shape or self._t_buf.dtype != tt.dtype or self._t_buf.device != tt.device
        )
        if need_new:
            self._x_buf = torch.empty_like(xt, memory_format=torch.contiguous_format)
            self._t_buf = torch.empty_like(tt, memory_format=torch.contiguous_format)
            self._log(f"[AICF] allocate capture buffers: x={tuple(xt.shape)} t={tuple(tt.shape)}")

        self._x_buf.copy_(xt)
        self._t_buf.copy_(tt)
        return (_wrap(self._x_buf), _wrap(self._t_buf))

    def bind_batch(self, batch) -> None:
        if self._x_buf is None or self._t_buf is None:
            raise RuntimeError("bind_batch called before prepare_capture_batch")

        x, t = batch
        xt = _unwrap(x)
        tt = _unwrap(t)

        if xt.shape != self._x_buf.shape or tt.shape != self._t_buf.shape:
            raise RuntimeError(
                f"bind_batch shape mismatch: got x={tuple(xt.shape)} t={tuple(tt.shape)} "
                f"expected x={tuple(self._x_buf.shape)} t={tuple(self._t_buf.shape)}"
            )
        self._x_buf.copy_(xt)
        self._t_buf.copy_(tt)

    def capture_begin(self) -> None:
        if not self.enabled:
            raise RuntimeError("capture_begin called but AICF backend is disabled")
        if self._capturing:
            raise RuntimeError("capture_begin called while already capturing")

        C.capture_reset()
        C.capture_begin()
        self._capturing = True
        self._captured = False

    def capture_end(self) -> None:
        if not self._capturing:
            raise RuntimeError("capture_end called but not capturing")
        C.capture_end()
        self._capturing = False
        self._captured = True

    def replay(self):
        if not self._captured:
            raise RuntimeError("replay called but no captured graph exists")
        C.replay()
        return None

    # -------------------------
    # Unified op_call wrappers
    # -------------------------
    def _op_call(self, kind, inputs, outputs, attrs: Optional[Dict[str, Any]] = None) -> None:
        if attrs is None:
            attrs = {}
        C.op_call(kind, inputs, outputs, attrs)

    # -------------------------
    # Ops
    # -------------------------
    def add(self, a: Tensor, b: Tensor) -> Tensor:
        if (not self.enabled):
            return self.fallback.add(a, b)

        ta, tb = _unwrap(a), _unwrap(b)
        if not (_is_cuda_contig_f32(ta) and _is_cuda_contig_f32(tb)):
            return self.fallback.add(a, b)
        if ta.numel() != tb.numel():
            return self.fallback.add(a, b)

        out = torch.empty_like(ta)
        self._op_call(C.OpKind.EltwiseAdd, [ta, tb], [out], {})
        return _wrap(out)

    def relu(self, x: Tensor) -> Tensor:
        if (not self.enabled):
            return self.fallback.relu(x)

        tx = _unwrap(x)
        if not _is_cuda_contig_f32(tx):
            return self.fallback.relu(x)

        out = torch.empty_like(tx)
        self._op_call(C.OpKind.EltwiseRelu, [tx], [out], {})
        return _wrap(out)

    def gemm(
        self,
        a: Tensor,
        b: Tensor,
        bias: Optional[Tensor] = None,
        act: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        # v0: bias/act는 아직 torch fallback (너 registry 확장되면 여기서 attrs로 내리면 됨)
        if (not self.enabled):
            return self.fallback.gemm(a, b, bias=bias, act=act, attrs=attrs)
        if bias is not None or act is not None:
            return self.fallback.gemm(a, b, bias=bias, act=act, attrs=attrs)

        ta, tb = _unwrap(a), _unwrap(b)
        if not (_is_cuda_contig_f32(ta) and _is_cuda_contig_f32(tb)):
            return self.fallback.gemm(a, b, bias=bias, act=act, attrs=attrs)
        if ta.ndim != 2 or tb.ndim != 2:
            return self.fallback.gemm(a, b, bias=bias, act=act, attrs=attrs)
        M, K = ta.shape
        K2, N = tb.shape
        if K2 != K:
            return self.fallback.gemm(a, b, bias=bias, act=act, attrs=attrs)

        out = torch.empty((M, N), device=ta.device, dtype=ta.dtype)
        self._op_call(C.OpKind.Gemm, [ta, tb], [out], attrs or {})
        return _wrap(out)

    def mse(self, y: Tensor, t: Tensor) -> Tensor:
        # mse는 아직 AICF op가 없으면 torch로
        return self.fallback.mse(y, t)
