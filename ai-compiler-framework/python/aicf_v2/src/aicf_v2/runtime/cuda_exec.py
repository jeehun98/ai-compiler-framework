# python/aicf_v2/src/aicf_v2/runtime/cuda_exec.py

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, List, Set

import torch

# 순환 참조 방지: TYPE_CHECKING 일 때만 Model 임포트
if TYPE_CHECKING:
    from ..model import Model

from ..builder import Builder
from ..compile.types import CompiledProgram
from ..graph import Op
from ..backends.cuda.registry import CudaRegistry
from ..backends.cuda.bridge import op_call, current_stream_u64
from .alloc import bind_and_alloc_slots
from .graph_capture import GraphCapturedProgram, capture_cuda_graph, replay_cuda_graph


def _role_vids(b: Builder, role: str) -> List[int]:
    """Builder에서 특정 역할의 vid 리스트를 추출하는 헬퍼 함수"""
    if role == "input":
        return list(getattr(b, "input_vids", []))
    if role == "param":
        return list(getattr(b, "param_vids", []))
    if role == "state":
        return list(getattr(b, "state_vids", []))
    return []


def _feed_signature(b: Builder, feed: Dict[str, torch.Tensor], copy_names: Set[str]) -> Tuple:
    """실제로 복사될(copy_roles) 텐서들에 대해서만 시그니처를 생성합니다."""
    items = []
    for name in copy_names:
        if name not in feed:
            raise KeyError(f"Missing feed for required input '{name}'")
        t = feed[name]
        items.append((name, (tuple(t.shape), str(t.dtype), str(t.device))))
    items.sort(key=lambda x: x[0])
    return tuple(items)


def _apply_abi_fixups(op: Op, ins: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    ABI fixups: 스칼라(rank-1, numel-1) 입력을 백엔드가 기대하는 rank-0으로 변환합니다.
    """
    hints = getattr(op, "hints", None) or {}
    idxs = hints.get("view_rank0_inputs", None)
    if idxs:
        for i in idxs:
            if 0 <= i < len(ins):
                t = ins[i]
                if t.dim() == 1 and t.numel() == 1:
                    ins[i] = t.view(())
    return ins


class CudaExecutor:
    def __init__(self, registry: Optional[CudaRegistry] = None):
        self.registry = registry or CudaRegistry()
        self._compiled_cache: Dict[int, CompiledProgram] = {}
        self._graph_cache: Dict[Any, GraphCapturedProgram] = {}

    def compile(self, m: Model) -> CompiledProgram:
        """모델을 IR 수준에서 컴파일합니다. (런타임 임포트로 순환 참조 해결)"""
        from ..compile.compile import compile_cuda
        return compile_cuda(m, self.registry)

    def compile_cached(self, m: Model) -> CompiledProgram:
        key = id(m.b)
        prog = self._compiled_cache.get(key)
        if prog is None:
            prog = self.compile(m)
            self._compiled_cache[key] = prog
        return prog

    def capture_prebuilt(
        self,
        m: Model,
        prog: CompiledProgram,
        sample_feed: Dict[str, torch.Tensor],
        *,
        mode: str = "train",
        warmup: int = 1,
    ) -> GraphCapturedProgram:
        """
        [Pre-capture] 실제 실행 루프 진입 전, 컴파일 단계에서 CUDA Graph를 미리 캡처합니다.
        """
        static_roles, copy_roles = self._get_roles(mode)
        key = self._cache_key(
            m, prog, sample_feed,
            mode=mode,
            static_roles=static_roles,
            copy_roles=copy_roles,
        )

        if key not in self._graph_cache:
            gprog = capture_cuda_graph(
                m, prog, sample_feed,
                stream=None,
                warmup=warmup,
                static_roles=static_roles,
                copy_roles=copy_roles,
            )
            self._graph_cache[key] = gprog

        return self._graph_cache[key]

    def _get_roles(self, mode: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """모드에 따른 메모리 고정/복사 정책 결정"""
        if mode == "train":
            # 학습 모드: 가중치(param)와 상태(state)의 GPU 주소를 고정하여 보존합니다.
            return ("input", "param", "state"), ("input",)
        # 추론 모드: 파라미터만 고정합니다.
        return ("input", "param"), ("input",)

    def _cache_key(
        self,
        m: Model,
        prog: CompiledProgram,
        feed: Dict[str, torch.Tensor],
        *,
        mode: str,
        static_roles: Tuple[str, ...],
        copy_roles: Tuple[str, ...],
    ):
        b = m.b
        plan_key = getattr(prog.plan, "plan_id", None) or id(prog.plan)

        # 이번 모드에서 실제로 외부에서 주입(copy)받아야 할 이름들 추출
        copy_names: Set[str] = set()
        for r in copy_roles:
            for vid in _role_vids(b, r):
                copy_names.add(b.values[vid].name)

        # copy_names에 대해서만 피드 체크 수행 (w, m, v 등 static은 무시)
        return (mode, plan_key, static_roles, copy_roles, _feed_signature(b, feed, copy_names))

    def run_compiled(
        self,
        m: Model,
        prog: CompiledProgram,
        feed: Dict[str, torch.Tensor],
        *,
        stream: Optional[int] = None,
        use_cuda_graph: bool = True,
        mode: str = "inference",
        warmup: int = 2,
    ) -> Dict[str, torch.Tensor]:
        b: Builder = m.b
        plan = prog.plan

        # -------------------------
        # 1. CUDA Graph Path
        # -------------------------
        if use_cuda_graph:
            if stream is not None:
                raise NotImplementedError("CUDA Graph with explicit stream is not supported yet")

            static_roles, copy_roles = self._get_roles(mode)
            key = self._cache_key(
                m, prog, feed,
                mode=mode,
                static_roles=static_roles,
                copy_roles=copy_roles,
            )

            gprog = self._graph_cache.get(key)
            if gprog is None:
                # 미리 캡처되지 않은 경우 실행 시점에 캡처를 수행합니다.
                # run_compiled의 warmup 값을 반영합니다.
                gprog = self.capture_prebuilt(m, prog, feed, mode=mode, warmup=warmup)

            return replay_cuda_graph(m, gprog, feed)

        # -------------------------
        # 2. Eager Path (Fallback)
        # -------------------------
        stream_u64 = int(stream) if stream is not None else current_stream_u64()
        slots = bind_and_alloc_slots(b, feed)

        # In-place 최적화 적용
        for out_vid, in_vid in plan.alias.items():
            slots[out_vid] = slots[in_vid]

        # 연산 실행
        for op in plan.ops:
            if getattr(op, "kind_id", None) is None:
                raise ValueError(f"[CudaExecutor] missing op.kind_id for {op.kind}")

            ins = _apply_abi_fixups(op, [slots[v] for v in op.inputs])
            outs = [slots[v] for v in op.outputs]

            op_call(
                int(op.kind_id),
                ins,
                outs,
                int(op.attr_schema),
                op.attr_blob,
                stream=stream_u64,
            )

        return self._extract_outputs(b, slots)

    def _extract_outputs(self, b: Builder, slots: Dict[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """최종 Vid 슬롯에서 출력 명칭에 맞춰 텐서들을 추출합니다."""
        out: Dict[str, torch.Tensor] = {}
        output_map = getattr(b, "outputs", None)
        if output_map:
            for name, vid in output_map.items():
                out[name] = slots[vid]
        else:
            for vid in b.output_vids:
                out[b.values[vid].name] = slots[vid]
        return out