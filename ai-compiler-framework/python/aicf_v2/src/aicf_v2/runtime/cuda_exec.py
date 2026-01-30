from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch

from ..model import Model
from ..tensor_spec import TensorSpec
from ..graph import Op, Value
from ..backends.cuda.registry import CudaRegistry
from ..backends.cuda.attrs import pack_attrs
from ..backends.cuda.bridge import op_call, current_stream_u64

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

def _alloc_from_spec(spec: TensorSpec) -> torch.Tensor:
    dt = _torch_dtype(spec.dtype)
    dev = torch.device(spec.device)
    return torch.empty(spec.shape, device=dev, dtype=dt)

@dataclass
class LoweredOp:
    kind: str
    kind_id: int
    attr_schema: int
    attr_blob: bytes
    in_vids: List[int]
    out_vids: List[int]

class CudaExecutor:
    def __init__(self, registry: Optional[CudaRegistry] = None):
        self.registry = registry or CudaRegistry()

    def lower(self, m: Model) -> List[LoweredOp]:
        """
        IR Op 스트림을 그대로 lowered로 변환.
        (rewrite/pass 없음)
        """
        b = m.b
        lowered: List[LoweredOp] = []
        for op in b.ops:
            ks = self.registry.lookup(op.kind)
            # runtime_flags는 실행 시점에 결정될 수 있으므로 여기서는 비워두고,
            # 최종 attr_blob도 여기서 만든다고 가정(현재는 inplace 안 씀)
            attr_blob = pack_attrs(op.kind, op.attrs, runtime_flags={"inplace": False})
            lowered.append(LoweredOp(
                kind=op.kind,
                kind_id=ks.kind_id,
                attr_schema=ks.attr_schema,
                attr_blob=attr_blob,
                in_vids=list(op.inputs),
                out_vids=list(op.outputs),
            ))
        return lowered

    def run(self, m: Model, feed: Dict[str, torch.Tensor], *, stream: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        feed: { value_name: torch.Tensor }  (inputs + params를 이름으로 공급)
        return: { output_name: torch.Tensor } (현재는 output alias를 별도 저장 안하니까 value.name 기준 반환)
        """
        b = m.b
        stream_u64 = int(stream) if stream is not None else current_stream_u64()

        # 1) vid -> tensor slot 준비
        slots: List[Optional[torch.Tensor]] = [None] * len(b.values)

        # 2) inputs/params 주입 (name 매칭)
        for vid in b.input_vids:
            name = b.values[vid].name
            if name not in feed:
                raise KeyError(f"Missing feed for input/param '{name}'")
            t = feed[name]
            # 기본 체크 (shape/dtype/device)
            spec = b.values[vid].spec
            if tuple(t.shape) != tuple(spec.shape):
                raise ValueError(f"Feed shape mismatch for '{name}': got {tuple(t.shape)} expected {spec.shape}")
            spec_dev = spec.device
            t_dev = t.device

            # "cuda" vs "cuda:0" 같은 케이스 허용
            if spec_dev == "cuda":
                if t_dev.type != "cuda":
                    raise ValueError(f"Feed device mismatch for '{name}': got {t_dev} expected cuda")
            elif spec_dev == "cpu":
                if t_dev.type != "cpu":
                    raise ValueError(f"Feed device mismatch for '{name}': got {t_dev} expected cpu")
            else:
                # 혹시 "cuda:0" 같은 형태를 spec에 넣는 방식도 허용하려면:
                if str(t_dev) != spec_dev:
                    raise ValueError(f"Feed device mismatch for '{name}': got {t_dev} expected {spec_dev}")

            slots[vid] = t

        # 3) 중간/출력 텐서 allocate (아직 메모리 플래닝 없음)
        for v in b.values:
            if slots[v.vid] is None:
                slots[v.vid] = _alloc_from_spec(v.spec)

        # 4) lower + execute
        lowered = self.lower(m)

        for lop in lowered:
            ins = [slots[v] for v in lop.in_vids]
            outs = [slots[v] for v in lop.out_vids]
            assert all(isinstance(t, torch.Tensor) for t in ins)
            assert all(isinstance(t, torch.Tensor) for t in outs)

            op_call(
                lop.kind_id,
                ins, outs,
                lop.attr_schema,
                lop.attr_blob,
                stream=stream_u64,
            )

        # 5) outputs 반환: model.b.output_vids 기준
        out: Dict[str, torch.Tensor] = {}
        for vid in b.output_vids:
            name = b.values[vid].name
            out[name] = slots[vid]  # type: ignore
        return out
