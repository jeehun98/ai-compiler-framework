# aicf_fw/fw/executor_fw.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from aicf_fw.fw.emit_ctx import FrozenGraph
from aicf_fw.fw.plan_fw import BindingPlanFW, bind_tensors_fw


@dataclass
class TraceItemFW:
    kid: str
    kind: str


class ExecutorFW:
    def __init__(self, *, graph: FrozenGraph, plan: BindingPlanFW):
        self.graph = graph
        self.plan = plan
        self._trace: List[TraceItemFW] = []
        self._trace_enabled = False

        # CUDA Graph (optional)
        self._captured = False
        self._cg = None

    def trace_reset(self):
        self._trace = []

    def trace_enable(self, b: bool):
        self._trace_enabled = bool(b)

    def trace_get(self):
        return list(self._trace)

    def _alloc_tmps(self, vmap: Dict[int, Any]) -> None:
        """
        tmp value에 대해 torch.empty를 만들어준다.
        """
        for vid, vdesc in enumerate(self.plan.values):
            if vdesc.role == "tmp":
                if vmap[vid] is None:
                    vmap[vid] = torch.empty(vdesc.shape, device=vdesc.device, dtype=vdesc.dtype)

    def _launch_one(self, op, vmap: Dict[int, Any]):
        kid = op.kernel_id
        if kid is None:
            raise RuntimeError(f"[ExecutorFW] missing kernel_id for op={op.op_kind}")

        # --- collect tensors ---
        in_tensors = [vmap[i] for i in op.inputs]
        out_tensors = [vmap[o] for o in op.outputs]

        # --- DISPATCH ---
        # TODO: 여기 한 줄을 네 backend/cuda.py 실제 API로 맞춰야 함.
        # 예시: from aicf_fw.backend.cuda import launch
        # launch(kid, *in_tensors, *out_tensors, **op.attrs)

        from aicf_fw.backend.cuda import launch  # <= 네 코드에 이 함수가 없으면 이름만 바꿔
        launch(kid, inputs=in_tensors, outputs=out_tensors, attrs=op.attrs)

        if self._trace_enabled:
            self._trace.append(TraceItemFW(kid=kid, kind=op.op_kind))

    def run(self, *, inputs: Dict[str, Any], params: Dict[str, Any], statics: Dict[str, Any], meta: Dict[str, Any] | None = None):
        vmap = bind_tensors_fw(self.plan, inputs=inputs, params=params, statics=statics, meta=meta)
        self._alloc_tmps(vmap)

        for op in self.graph.ops:
            self._launch_one(op, vmap)

    # ---- CUDA Graph capture/replay (optional minimal) ----
    def capture(self, *, inputs: Dict[str, Any], params: Dict[str, Any], statics: Dict[str, Any], meta: Dict[str, Any] | None = None):
        # 단순 안정성: 한번 dry-run 해서 tmp alloc 완료
        self.run(inputs=inputs, params=params, statics=statics, meta=meta)
        torch.cuda.synchronize()

        stream = torch.cuda.current_stream()
        self._cg = torch.cuda.CUDAGraph()
        self._captured = False

        # 캡처는 동일 포인터가 필요: vmap을 유지해야 진짜 의미 있음.
        # 초미니에선 capture API만 “형태”로 만들고,
        # 실제 pointer-stable vmap 고정은 다음 단계에서 한다.
        self._cg.capture_begin()
        self.run(inputs=inputs, params=params, statics=statics, meta=meta)
        self._cg.capture_end()

        self._captured = True

    def replay(self, n: int = 1, sync: bool = False):
        if not self._captured or self._cg is None:
            raise RuntimeError("graph is not captured")
        for _ in range(int(n)):
            self._cg.replay()
        if sync:
            torch.cuda.synchronize()
