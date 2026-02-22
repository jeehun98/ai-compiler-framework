# python/aicf_v2/src/aicf_v2/runtime/graph_capture.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

import torch

# 순환 참조 방지: TYPE_CHECKING 일 때만 Model 임포트
if TYPE_CHECKING:
    from ..model import Model

from ..builder import Builder
from ..compile.types import CompiledProgram
from ..graph import Op
from ..backends.cuda.bridge import op_call, current_stream_u64
from .alloc import alloc_from_spec, _device_ok, _torch_dtype


# -----------------------------
# helpers
# -----------------------------
def _check_feed_tensor(
    name: str,
    spec,
    t: torch.Tensor,
    *,
    require_contiguous: bool = True,
) -> torch.Tensor:
    if tuple(t.shape) != tuple(spec.shape):
        raise ValueError(f"Feed shape mismatch for '{name}': got {tuple(t.shape)} expected {spec.shape}")

    if not _device_ok(spec.device, t):
        raise ValueError(f"Feed device mismatch for '{name}': got {t.device} expected {spec.device}")

    exp = _torch_dtype(spec.dtype)
    if t.dtype != exp:
        raise ValueError(f"Feed dtype mismatch for '{name}': got {t.dtype} expected {exp}")

    # Graph path에서는 숨은 할당을 없애기 위해 contiguous를 강제 요구하는 편이 안전합니다.
    # (캡처 시점에는 완화 가능하지만, 기본은 엄격하게 둡니다.)
    if require_contiguous and (not t.is_contiguous()):
        raise ValueError(f"Feed tensor for '{name}' must be contiguous for CUDA Graph path.")
    if (not require_contiguous) and (not t.is_contiguous()):
        t = t.contiguous()

    return t


def _role_vids(b: Builder, role: str) -> List[int]:
    if role == "input":
        return list(getattr(b, "input_vids", []))
    if role == "param":
        return list(getattr(b, "param_vids", []))
    if role == "state":
        return list(getattr(b, "state_vids", []))
    raise KeyError(f"unknown role '{role}'")


def _all_external_vids(b: Builder) -> List[int]:
    return list(getattr(b, "external_vids", []))


def _apply_abi_fixups(op: Op, ins: list[torch.Tensor]) -> list[torch.Tensor]:
    hints = getattr(op, "hints", None) or {}
    idxs = hints.get("view_rank0_inputs", None)
    if idxs:
        for i in idxs:
            if 0 <= i < len(ins):
                t = ins[i]
                if t.dim() == 1 and t.numel() == 1:
                    ins[i] = t.view(())
    return ins


# -----------------------------
# captured program container
# -----------------------------
@dataclass
class GraphCapturedProgram:
    prog: CompiledProgram
    slots: List[torch.Tensor]                 # slots[vid] -> Tensor
    ext_bufs: Dict[int, torch.Tensor]         # external vid -> fixed buffer tensor
    graph: torch.cuda.CUDAGraph
    static_roles: Tuple[str, ...]
    copy_roles: Tuple[str, ...]


# -----------------------------
# core capture / replay
# -----------------------------
def capture_cuda_graph(
    m: Model,
    prog: CompiledProgram,
    feed: Dict[str, torch.Tensor],
    *,
    stream: Optional[int] = None,
    warmup: int = 2,
    static_roles: Iterable[str] = ("input", "param"),
    copy_roles: Iterable[str] = ("input",),
) -> GraphCapturedProgram:
    """
    고정 포인터 정책을 사용하여 CUDA Graph를 캡처합니다.
    """
    if stream is not None:
        raise NotImplementedError("capture uses current torch stream; stream=... not supported yet")

    b: Builder = m.b
    plan = prog.plan
    ops = plan.ops

    static_roles = tuple(static_roles)
    copy_roles = tuple(copy_roles)

    # 1) 모든 외부 변수(externals)는 static_roles에 포함되어야 함 (포인터 고정 보장)
    static_vids: List[int] = []
    for r in static_roles:
        static_vids += _role_vids(b, r)
    static_set = set(static_vids)

    ext_vids = _all_external_vids(b)
    missing = [vid for vid in ext_vids if vid not in static_set]
    if missing:
        names = [b.values[v].name for v in missing]
        raise ValueError(
            f"CUDA Graph requires fixed pointers for all externals. "
            f"Missing from static_roles: {names}  static_roles={static_roles}"
        )

    # 2) 메모리 슬롯 할당 (Fixed Buffers)
    # vid가 항상 0..N-1로 연속이라는 가정을 깨기 위해 max_vid 기반으로 슬롯 크기 결정
    max_vid = max(v.vid for v in b.values)
    slots: List[Optional[torch.Tensor]] = [None] * (max_vid + 1)
    ext_bufs: Dict[int, torch.Tensor] = {}

    # 외부(external) vid들은 그래프 수명 동안 고정될 주소 버퍼를 따로 할당
    # (캡처 시점엔 contiguous 완화 가능하지만, 기본적으로는 요구)
    for vid in ext_vids:
        v = b.values[vid]
        name = v.name
        if name not in feed:
            raise KeyError(f"Missing feed for external '{name}'")

        # 캡처 진입 단계에서는 사용자가 non-contig을 줘도 여기서 1회 contig로 정리 가능
        src = _check_feed_tensor(name, v.spec, feed[name], require_contiguous=False)

        buf = alloc_from_spec(v.spec)
        buf.copy_(src)
        slots[vid] = buf
        ext_bufs[vid] = buf

    # 내부 텐서 슬롯들도 모두 할당
    for v in b.values:
        if slots[v.vid] is None:
            slots[v.vid] = alloc_from_spec(v.spec)

    slots_t: List[torch.Tensor] = [t for t in slots if t is not None]  # type: ignore

    # 주의: 위 comprehension은 vid 인덱스 정렬을 깨므로, 반드시 원래 인덱스 유지 리스트로 변환해야 함
    # 따라서 올바른 변환을 다시 수행
    slots_t = [slots[i] for i in range(len(slots))]  # type: ignore

    # 3) Alias 결정 사항 적용 (In-place)
    for out_vid, in_vid in plan.alias.items():
        slots_t[out_vid] = slots_t[in_vid]
        if out_vid in ext_bufs:
            ext_bufs[out_vid] = slots_t[in_vid]

    # 4) Warmup (동일 버퍼를 사용한 사전 실행)
    for _ in range(int(warmup)):
        _copy_roles_into_ext_bufs(
            b, ext_bufs, feed,
            copy_roles=copy_roles,
            require_contiguous=True,   # graph path 성능/안정성 위해 엄격
        )
        _run_ops(ops, slots_t)

    # warmup 완료 대기 (필요)
    torch.cuda.synchronize()

    # 5) Graph Capture
    graph = torch.cuda.CUDAGraph()

    _copy_roles_into_ext_bufs(
        b, ext_bufs, feed,
        copy_roles=copy_roles,
        require_contiguous=True,
    )

    with torch.cuda.graph(graph):
        _run_ops(ops, slots_t)

    # 캡처 직후 sync는 보통 불필요하지만, 안정성/디버깅을 위해 유지하고 싶다면 켤 수 있음
    # torch.cuda.synchronize()

    return GraphCapturedProgram(
        prog=prog,
        slots=slots_t,
        ext_bufs=ext_bufs,
        graph=graph,
        static_roles=static_roles,
        copy_roles=copy_roles,
    )


def replay_cuda_graph(
    m: Model,
    gprog: GraphCapturedProgram,
    feed: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    캡처된 그래프를 실행(Replay)합니다.
    """
    b: Builder = m.b

    # copy_roles에 해당하는 텐서들만 내부 버퍼로 복사 (예: 새로운 입력 데이터)
    _copy_roles_into_ext_bufs(
        b, gprog.ext_bufs, feed,
        copy_roles=gprog.copy_roles,
        require_contiguous=True,  # replay 시점에는 숨은 할당 금지
    )

    # GPU 하드웨어 가속 실행
    gprog.graph.replay()

    # 출력 추출
    out: Dict[str, torch.Tensor] = {}
    output_map = getattr(b, "outputs", None)
    if output_map:
        for oname, vid in output_map.items():
            out[oname] = gprog.slots[vid]
    else:
        for vid in b.output_vids:
            out[b.values[vid].name] = gprog.slots[vid]
    return out


# -----------------------------
# internal ops
# -----------------------------
def _copy_roles_into_ext_bufs(
    b: Builder,
    ext_bufs: Dict[int, torch.Tensor],
    feed: Dict[str, torch.Tensor],
    *,
    copy_roles: Tuple[str, ...],
    require_contiguous: bool,
) -> None:
    copy_set = set(copy_roles)
    for role in ("input", "param", "state"):
        if role not in copy_set:
            continue
        for vid in _role_vids(b, role):
            v = b.values[vid]
            name = v.name
            if name not in feed:
                raise KeyError(f"Missing feed for '{name}' (role={role})")

            src = _check_feed_tensor(name, v.spec, feed[name], require_contiguous=require_contiguous)

            if vid not in ext_bufs:
                raise KeyError(f"internal error: ext buffer missing for vid={vid} name='{name}'")

            ext_bufs[vid].copy_(src)


def _run_ops(ops: List[Op], slots: List[torch.Tensor]) -> None:
    # 캡처/리플레이 모두 "현재 스트림"을 기준으로 호출하도록 매번 갱신
    stream_u64 = current_stream_u64()

    for op in ops:
        # 필수 캐시 체크 (Emitter가 채웠어야 함)
        if getattr(op, "kind_id", None) is None:
            raise ValueError(f"[graph_capture] missing op.kind_id (kind='{op.kind}', name='{op.name}')")
        if getattr(op, "attr_schema", None) is None:
            raise ValueError(f"[graph_capture] missing op.attr_schema (kind='{op.kind}', name='{op.name}')")
        if getattr(op, "attr_blob", None) is None:
            raise ValueError(f"[graph_capture] missing op.attr_blob (kind='{op.kind}', name='{op.name}')")

        ins = [slots[v] for v in op.inputs]
        outs = [slots[v] for v in op.outputs]

        # ABI fixups (hints 적용)
        ins = _apply_abi_fixups(op, ins)

        op_call(
            int(op.kind_id),
            ins,
            outs,
            int(op.attr_schema),
            op.attr_blob,
            stream=stream_u64,
        )