# examples/python/aicf_fw/aicf_backend.py
from __future__ import annotations
from typing import Optional, Any, Dict
import os
import torch

from .torch_backend import TorchBackend, BackendConfig
from .tensor import Tensor

# C++/CUDA 바인딩 모듈 (이름은 네가 실제로 만드는 pybind/ctypes 모듈명에 맞춰야 함)
# 예: tools/codegen로 만든다고 했으니 일단 aicf_cuda 라고 가정
try:
    import aicf_cuda  # <- 너의 파이썬 확장 모듈명으로 바꿔
except Exception:
    aicf_cuda = None


def _is_cuda_f32_contig(x: Tensor) -> bool:
    t = x.t  # Tensor 래퍼가 torch.Tensor를 .t로 들고 있다는 가정 (아니면 맞춰)
    return t.is_cuda and t.dtype == torch.float32 and t.is_contiguous()


class AicfBackend:
    """
    AICF backend (partial).
    - add/relu/gemm만 AICF 커널로 점진 교체
    - 조건 불만족 또는 바인딩 미존재 시 TorchBackend로 fallback
    """

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg
        self.fallback = TorchBackend(cfg)

        self.enabled = (os.environ.get("AICF_ENABLE_AICF_KERNELS", "1") == "1")
        self.debug = (os.environ.get("AICF_BACKEND_DEBUG", "0") == "1")

        if aicf_cuda is None:
            if self.debug:
                print("[AICF] aicf_cuda python binding not found -> fallback to torch")
            self.enabled = False

    # ---- utility ----
    def _log(self, msg: str):
        if self.debug:
            print(msg)

    # ---- ops ----
    def add(self, a: Tensor, b: Tensor) -> Tensor:
        # AICF add: 1D contiguous F32
        if not self.enabled:
            return self.fallback.add(a, b)

        if not (_is_cuda_f32_contig(a) and _is_cuda_f32_contig(b)):
            return self.fallback.add(a, b)

        ta, tb = a.t, b.t
        if ta.numel() != tb.numel():
            return self.fallback.add(a, b)

        # 현재 C++ add_f32는 1D만 명시했으니 flatten해서 넘김
        out = torch.empty_like(ta)
        ok = aicf_cuda.add_f32(ta, tb, out)  # 바인딩 시그니처에 맞춰야 함
        if not ok:
            self._log("[AICF] add_f32 failed -> fallback")
            return self.fallback.add(a, b)

        return Tensor(out)

    def relu(self, x: Tensor) -> Tensor:
        if not self.enabled:
            return self.fallback.relu(x)

        if not _is_cuda_f32_contig(x):
            return self.fallback.relu(x)

        tx = x.t
        out = torch.empty_like(tx)
        ok = aicf_cuda.relu_f32(tx, out)
        if not ok:
            self._log("[AICF] relu_f32 failed -> fallback")
            return self.fallback.relu(x)

        return Tensor(out)

    def gemm(self, a: Tensor, b: Tensor, bias: Optional[Tensor] = None,
             act: Optional[str] = None, attrs: Optional[Dict[str, Any]] = None) -> Tensor:
        # 현재 C++ gemm_f32는 bias/act/attrs 미지원이므로 조건을 빡세게 걸고 fallback
        if not self.enabled:
            return self.fallback.gemm(a, b, bias=bias, act=act, attrs=attrs)

        if bias is not None or act is not None:
            return self.fallback.gemm(a, b, bias=bias, act=act, attrs=attrs)

        if not (_is_cuda_f32_contig(a) and _is_cuda_f32_contig(b)):
            return self.fallback.gemm(a, b, bias=bias, act=act, attrs=attrs)

        ta, tb = a.t, b.t
        if ta.ndim != 2 or tb.ndim != 2:
            return self.fallback.gemm(a, b, bias=bias, act=act, attrs=attrs)

        M, K = ta.shape
        K2, N = tb.shape
        if K2 != K:
            return self.fallback.gemm(a, b, bias=bias, act=act, attrs=attrs)

        out = torch.empty((M, N), device=ta.device, dtype=ta.dtype)
        ok = aicf_cuda.gemm_f32(ta, tb, out)  # 바인딩 시그니처에 맞춰야 함
        if not ok:
            self._log("[AICF] gemm_f32 failed -> fallback")
            return self.fallback.gemm(a, b, bias=bias, act=act, attrs=attrs)

        return Tensor(out)

    # mse 같은 건 아직 torch로 유지
    def mse(self, y: Tensor, t: Tensor) -> Tensor:
        return self.fallback.mse(y, t)
