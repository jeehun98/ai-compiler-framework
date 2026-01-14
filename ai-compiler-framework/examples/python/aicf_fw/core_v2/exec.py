from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from aicf_fw.backend import get_backend
from .ir import IRGraph
from .plan import BindingPlan, allocate_static_env


@dataclass
class ExecOptions:
    debug: bool = False          # per-op IO ptr/shape 출력
    debug_limit: int = 50
    check_shapes: bool = True    # plan shape과 실제 tensor shape 검사
    check_device: bool = True    # cuda device 검사
    check_dtype: bool = True     # dtype 검사


def _tmeta(t: torch.Tensor) -> str:
    return f"ptr={t.data_ptr()} shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}"


def _unwrap(x) -> torch.Tensor:
    # torch.Tensor or wrapper with .data (ex: Parameter/Tensor wrapper)
    return x.data if hasattr(x, "data") else x


def _assert_tensor_matches(spec, t: torch.Tensor, opts: ExecOptions):
    if opts.check_shapes and tuple(t.shape) != tuple(spec.shape):
        raise RuntimeError(f"[plan] shape mismatch for vid{spec.vid}({spec.name}): {tuple(t.shape)} != {tuple(spec.shape)}")
    if opts.check_dtype:
        # spec.dtype is string like "torch.float32" or "float32" depending on your IR printer
        # We'll compare via substring.
        ds = str(t.dtype)
        if ("float16" in spec.dtype and "float16" not in ds) or \
           ("bfloat16" in spec.dtype and "bfloat16" not in ds) or \
           ("float32" in spec.dtype and "float32" not in ds) or \
           ("float64" in spec.dtype and "float64" not in ds) or \
           ("int32" in spec.dtype and "int32" not in ds) or \
           ("int64" in spec.dtype and "int64" not in ds):
            raise RuntimeError(f"[plan] dtype mismatch for vid{spec.vid}({spec.name}): {t.dtype} vs {spec.dtype}")
    if opts.check_device:
        if "cuda" in spec.device:
            if not t.is_cuda:
                raise RuntimeError(f"[plan] device mismatch for vid{spec.vid}({spec.name}): expected cuda, got {t.device}")


class PlannedExecutor:
    """
    Stage4: plan 기반 실행기
      - env는 (static alloc) + (user inputs) + (user params)로만 구성
      - lowered를 순서대로 backend.op_call_out로 실행
      - in-place(out==in) op도 lowered대로 그대로 수행

    NOTE:
      - 여기서는 IRExecutor처럼 SSA rebind를 할 필요가 없음.
        왜냐면 lowered에서 vid는 'storage'로 해석되고, plan이 storage를 고정하기 때문.
      - 다만, 안전하게 outputs가 다른 텐서를 가리키는 경우(env[vid]=out)을 허용하면 확장에 유리함.
    """

    def __init__(
        self,
        *,
        ir: IRGraph,
        lowered: List[Dict[str, Any]],
        plan: BindingPlan,
        backend: Any = None,
        device: Optional[torch.device] = None,
        opts: Optional[ExecOptions] = None,
    ):
        self.ir = ir
        self.lowered = lowered
        self.plan = plan
        self.backend = backend
        self.device = device
        self.opts = opts or ExecOptions()

        # runtime env (built per run)
        self.env: Dict[int, torch.Tensor] = {}

    def build_env(
        self,
        *,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        reuse_static: bool = False,
    ) -> Dict[int, torch.Tensor]:
        """
        inputs: {"x": torch.Tensor, "t": torch.Tensor, ...}
        params: {"0.W": torch.Tensor, "0.b": torch.Tensor, ...}
        """
        opts = self.opts

        # 1) statics allocate
        if (not reuse_static) or (not self.env):
            self.env = allocate_static_env(self.ir, self.plan, device=self.device)

        # 2) bind inputs/params by name using plan.specs
        name_to_vid = {s.name: int(s.vid) for s in self.plan.specs.values()}

        def bind_by_name(src: Dict[str, Any], kind: str):
            for name, obj in src.items():
                if name not in name_to_vid:
                    raise KeyError(f"[plan] unknown {kind} name='{name}'. known={list(name_to_vid.keys())}")
                vid = name_to_vid[name]
                spec = self.plan.specs[vid]
                t = _unwrap(obj)
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"[plan] {kind} '{name}' must be torch.Tensor-like, got {type(obj)}")

                _assert_tensor_matches(spec, t, opts)
                self.env[int(vid)] = t

        bind_by_name(inputs, "input")
        bind_by_name(params, "param")

        # 3) sanity: ensure all required vids exist
        for vid in self.plan.inputs + self.plan.params + self.plan.statics:
            if int(vid) not in self.env:
                spec = self.plan.specs[int(vid)]
                raise RuntimeError(f"[plan] env missing vid{vid} ({spec.name}) role={spec.role}")

        return self.env

    @torch.no_grad()
    def run(
        self,
        *,
        inputs: Dict[str, Any],
        params: Dict[str, Any],
        reuse_static: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Execute lowered ops. Returns env.
        """
        bk = self.backend or get_backend()
        self.build_env(inputs=inputs, params=params, reuse_static=reuse_static)

        dbg = self.opts.debug
        limit = int(self.opts.debug_limit)

        for i, it in enumerate(self.lowered):
            op = str(it["op"])
            in_vids = [int(x) for x in it.get("inputs", [])]
            out_vids = [int(y) for y in it.get("outputs", [])]
            attrs = dict(it.get("attrs", {}) or {})

            ins_t = [self.env[v] for v in in_vids]
            outs_t = [self.env[v] for v in out_vids]

            if dbg and i < limit:
                ins_s = ", ".join([f"v{v:03d}:{_tmeta(self.env[v])}" for v in in_vids])
                outs_s = ", ".join([f"v{v:03d}:{_tmeta(self.env[v])}" for v in out_vids])
                print(f"[exec][#{i:03d}] {op} attrs={attrs}")
                print(f"  in : {ins_s}")
                print(f"  out: {outs_s}")

            # dispatch
            bk.op_call_out(op, ins_t, outs_t, attrs)

            # allow out tensors to change identity (future-proof)
            for v, t in zip(out_vids, outs_t):
                self.env[int(v)] = t

        return self.env
