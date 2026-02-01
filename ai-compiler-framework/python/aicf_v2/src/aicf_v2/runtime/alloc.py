from __future__ import annotations
from typing import Dict, List, Optional, Set, Iterable

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


def _check_feed_tensor(name: str, spec: TensorSpec, t: torch.Tensor) -> torch.Tensor:
    if tuple(t.shape) != tuple(spec.shape):
        raise ValueError(f"Feed shape mismatch for '{name}': got {tuple(t.shape)} expected {spec.shape}")

    if not _device_ok(spec.device, t):
        raise ValueError(f"Feed device mismatch for '{name}': got {t.device} expected {spec.device}")

    exp = _torch_dtype(spec.dtype)
    if t.dtype != exp:
        raise ValueError(f"Feed dtype mismatch for '{name}': got {t.dtype} expected {exp}")

    if not t.is_contiguous():
        t = t.contiguous()

    return t


def bind_and_alloc_slots(
    b: Builder,
    feed: Dict[str, torch.Tensor],
    *,
    static_roles: Iterable[str] = ("input",),
) -> List[torch.Tensor]:
    """
    slots[vid] = Tensor
      - externals(=b.external_vids) bind/copy from feed
      - alloc remaining by spec

    static_roles: 해당 role("input"/"param"/"state")의 external은
                  slots에 고정 버퍼를 만들고 feed 값을 copy_ (CUDA Graph 캡처용)
    """
    static_role_set: Set[str] = set(static_roles)

    slots: List[Optional[torch.Tensor]] = [None] * len(b.values)

    # bind/copy externals
    for vid in b.external_vids:
        v = b.values[vid]
        name = v.name
        spec = v.spec
        role = v.role

        if name not in feed:
            raise KeyError(f"Missing feed for external '{name}'")

        t = _check_feed_tensor(name, spec, feed[name])

        if role in static_role_set:
            buf = alloc_from_spec(spec)
            buf.copy_(t)
            slots[vid] = buf
        else:
            slots[vid] = t

    # alloc remaining (temps, intermediates, outputs)
    for v in b.values:
        if slots[v.vid] is None:
            slots[v.vid] = alloc_from_spec(v.spec)

    return [t for t in slots if t is not None]


def copy_feed_into_slots(
    b: Builder,
    slots: List[torch.Tensor],
    feed: Dict[str, torch.Tensor],
    *,
    copy_roles: Iterable[str] = ("input",),
) -> None:
    """
    replay 때 값 갱신.

    ✅ 중요한 변경:
    - train/inference 상관없이, "순수 입력"은 b.input_vids로 정의한다.
    - 따라서 replay copy는 기본적으로 b.input_vids만 수행하면 됨.
    - (원하면 inference에서 param도 copy하고 싶을 때만 별도 옵션을 추가)
    """

    # 1) always copy real runtime inputs
    for vid in getattr(b, "input_vids", []):
        v = b.values[vid]
        name = v.name
        if name not in feed:
            raise KeyError(f"Missing feed for input '{name}'")
        src = _check_feed_tensor(name, v.spec, feed[name])
        slots[vid].copy_(src)

    # 2) optional: copy params too (inference weight hot-swap용)
    # copy_roles에 "param"이 들어온 경우에만 추가로 복사
    if "param" in set(copy_roles):
        for vid in getattr(b, "param_vids", []):
            v = b.values[vid]
            name = v.name
            if name not in feed:
                raise KeyError(f"Missing feed for param '{name}'")
            src = _check_feed_tensor(name, v.spec, feed[name])
            slots[vid].copy_(src)

    # state는 절대 replay에서 copy하지 않는다 (train 누적 업데이트 깨짐)
