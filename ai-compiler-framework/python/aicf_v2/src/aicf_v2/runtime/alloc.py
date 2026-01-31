from __future__ import annotations
from typing import Dict, List, Optional

import torch

from ..tensor_spec import TensorSpec
from ..builder import Builder

_TORCH_DTYPE = {
    "f16": torch.float16,
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "i32": torch.int32,
    "i64": torch.int64,
}


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype not in _TORCH_DTYPE:
        raise KeyError(f"unsupported dtype string '{dtype}'")
    return _TORCH_DTYPE[dtype]


def alloc_from_spec(spec: TensorSpec) -> torch.Tensor:
    dt = _torch_dtype(spec.dtype)
    dev = torch.device(spec.device)
    return torch.empty(spec.shape, device=dev, dtype=dt)


def _device_ok(spec_dev: str, t: torch.Tensor) -> bool:
    # allow "cuda" vs "cuda:0"
    if spec_dev == "cuda":
        return t.device.type == "cuda"
    if spec_dev == "cpu":
        return t.device.type == "cpu"
    return str(t.device) == spec_dev


def bind_and_alloc_slots(b: Builder, feed: Dict[str, torch.Tensor]) -> List[Optional[torch.Tensor]]:
    """
    slots[vid] = Tensor
      - bind feed for input_vids
      - alloc remaining by spec
    """
    slots: List[Optional[torch.Tensor]] = [None] * len(b.values)

    # bind inputs/params
    for vid in b.input_vids:
        name = b.values[vid].name
        if name not in feed:
            raise KeyError(f"Missing feed for input/param '{name}'")

        t = feed[name]
        spec = b.values[vid].spec

        if tuple(t.shape) != tuple(spec.shape):
            raise ValueError(f"Feed shape mismatch for '{name}': got {tuple(t.shape)} expected {spec.shape}")

        if not _device_ok(spec.device, t):
            raise ValueError(f"Feed device mismatch for '{name}': got {t.device} expected {spec.device}")

        # dtype check (원하면 완화 가능)
        exp = _torch_dtype(spec.dtype)
        if t.dtype != exp:
            raise ValueError(f"Feed dtype mismatch for '{name}': got {t.dtype} expected {exp}")

        if not t.is_contiguous():
            t = t.contiguous()

        slots[vid] = t

    # alloc remaining
    for v in b.values:
        if slots[v.vid] is None:
            slots[v.vid] = alloc_from_spec(v.spec)

    return slots
