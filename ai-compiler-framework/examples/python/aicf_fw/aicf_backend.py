# examples/python/aicf_fw/aicf_backend.py
from __future__ import annotations
from typing import Optional, Any, Dict, Tuple
import os
import torch

from .torch_backend import TorchBackend, BackendConfig
from .tensor import Tensor

# IMPORTANT: build/python/aicf_cuda/_C.pyd 구조면 보통 이게 맞음
try:
    from aicf_cuda import _C as aicf_cuda
except Exception:
    aicf_cuda = None


def _unwrap_torch(x: Any) -> torch.Tensor:
    return x.t if hasattr(x, "t") else x


def _wrap_tensor(x: torch.Tensor) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


class AicfBackend:
    """
    AICF backend (partial).
    - op_call 기반 일관성 유지
    - capture: "고정 input buffer"로 캡처하고, 매 step bind_batch에서 copy_로 갱신
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

        # fixed buffers for capture/replay
        self._x_buf: Optional[torch.Tensor] = None
        self._t_buf: Optional[torch.Tensor] = None

        if aicf_cuda is None:
            if self.debug:
                print("[AICF] aicf_cuda binding not found -> fallback to torch")
            self.enabled = False

    def _log(self, msg: str):
        if self.debug:
            print(msg)

    def _has(self, name: str) -> bool:
        return (aicf_cuda is not None) and hasattr(aicf_cuda, name)

    # -----------------------
    # Trainer-required methods
    # -----------------------
    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if hasattr(self.fallback, "set_mode"):
            self.fallback.set_mode(mode)

    def warmup(self, model, sample_batch) -> None:
        # 여기서는 특별히 할 일 없음 (Trainer warmup_steps가 실질 warmup)
        # 원하면 C++ 쪽 register/variant 선택 트리거를 여기서 할 수도 있음
        return

    def prepare_capture_batch(self, batch) -> Tuple[Any, Any]:
        """
        캡처/리플레이에서 사용할 "고정 입력 버퍼"를 만들고,
        Trainer가 캡처할 때는 이 버퍼를 입력으로 사용하게 함.

        반환: (x_wrapped, t_wrapped) - 네 framework Tensor 래퍼 형태로 돌려줌.
        """
        x, t = batch
        xt = _unwrap_torch(x)
        tt = _unwrap_torch(t)

        if not (isinstance(xt, torch.Tensor) and isinstance(tt, torch.Tensor)):
            raise RuntimeError("prepare_capture_batch expects torch.Tensor or Tensor wrapper with .t")

        if not xt.is_cuda or not tt.is_cuda:
            raise RuntimeError("capture mode requires CUDA tensors")

        # 버퍼는 "캡처 시점에 고정되는 주소"여야 하므로 한번 만들면 재사용.
        # shape/dtype/device가 바뀌면 다시 만들어야 함.
        need_new = (
            self._x_buf is None or self._t_buf is None
            or self._x_buf.shape != xt.shape or self._x_buf.dtype != xt.dtype or self._x_buf.device != xt.device
            or self._t_buf.shape != tt.shape or self._t_buf.dtype != tt.dtype or self._t_buf.device != tt.device
        )

        if need_new:
            # contiguous 고정 (캡처 안정성 + 커널 제약)
            self._x_buf = torch.empty_like(xt, memory_format=torch.contiguous_format)
            self._t_buf = torch.empty_like(tt, memory_format=torch.contiguous_format)
            self._log(f"[AICF] allocate capture buffers: x={tuple(xt.shape)} t={tuple(tt.shape)}")

        # 첫 데이터 주입
        self._x_buf.copy_(xt)
        self._t_buf.copy_(tt)

        return (_wrap_tensor(self._x_buf), _wrap_tensor(self._t_buf))

    def bind_batch(self, batch) -> None:
        """
        매 step 입력 갱신: 포인터를 바꾸지 않고, 고정 버퍼에 copy_.
        """
        if self._x_buf is None or self._t_buf is None:
            raise RuntimeError("bind_batch called before prepare_capture_batch")

        x, t = batch
        xt = _unwrap_torch(x)
        tt = _unwrap_torch(t)

        # shape mismatch면 capture 자체가 다른 signature -> 다시 캡처해야 정상
        if xt.shape != self._x_buf.shape or tt.shape != self._t_buf.shape:
            raise RuntimeError(
                f"bind_batch shape mismatch: got x={tuple(xt.shape)} t={tuple(tt.shape)} "
                f"expected x={tuple(self._x_buf.shape)} t={tuple(self._t_buf.shape)}. "
                "Need a new capture/graph key."
            )

        self._x_buf.copy_(xt)
        self._t_buf.copy_(tt)

    def capture_begin(self) -> None:
        if not self._has("capture_begin"):
            raise RuntimeError("aicf_cuda.capture_begin not found (C++ binding missing)")

        if self._capturing:
            raise RuntimeError("capture_begin called while already capturing")

        # 기존 캡처 날림
        if self._has("capture_reset"):
            aicf_cuda.capture_reset()

        aicf_cuda.capture_begin()
        self._capturing = True
        self._captured = False

    def capture_end(self) -> None:
        if not self._has("capture_end"):
            raise RuntimeError("aicf_cuda.capture_end not found (C++ binding missing)")

        if not self._capturing:
            raise RuntimeError("capture_end called but not capturing")

        aicf_cuda.capture_end()
        self._capturing = False
        self._captured = True

    def replay(self):
        if not self._captured:
            raise RuntimeError("replay called but no captured graph exists")

        if not self._has("replay"):
            raise RuntimeError("aicf_cuda.replay not found (C++ binding missing)")

        # replay는 실행만. loss는 캡처된 그래프가 써놓는 텐서를 Trainer가 읽는 구조로 가는 게 안정적.
        aicf_cuda.replay()
        return None

    # -----------------------
    # Ops (네 기존 구현 유지 가능)
    # -----------------------
    def add(self, a: Tensor, b: Tensor) -> Tensor:
        return self.fallback.add(a, b)

    def relu(self, x: Tensor) -> Tensor:
        return self.fallback.relu(x)

    def gemm(self, a: Tensor, b: Tensor, bias: Optional[Tensor] = None,
             act: Optional[str] = None, attrs: Optional[Dict[str, Any]] = None) -> Tensor:
        return self.fallback.gemm(a, b, bias=bias, act=act, attrs=attrs)

    def mse(self, y: Tensor, t: Tensor) -> Tensor:
        return self.fallback.mse(y, t)
