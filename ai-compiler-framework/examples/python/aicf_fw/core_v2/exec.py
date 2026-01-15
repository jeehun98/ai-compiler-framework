# aicf_fw/core_v2/exec.py
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
        raise RuntimeError(
            f"[plan] shape mismatch for vid{spec.vid}({spec.name}): {tuple(t.shape)} != {tuple(spec.shape)}"
        )
    if opts.check_dtype:
        # spec.dtype is string like "torch.float32" or "float32"
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


def _canon_name(s: str) -> str:
    """
    BindingPlan spec.name을 사용자 params 키와 매칭하기 위한 canonicalization.
    예:
      "d_2.W" -> "2.W"
      "d_0.b" -> "0.b"
      "2.W"   -> "2.W"
    """
    s = str(s)
    if s.startswith("d_"):
        s = s[2:]
    return s


def _build_name_maps(plan: BindingPlan) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """
    plan.specs (vid->spec)에서:
      - exact name -> [vid...]
      - canon name -> [vid...]
    NOTE:
      - name 충돌이 있을 수 있으므로 list로 보관한다.
    """
    name_to_vids: Dict[str, List[int]] = {}
    canon_to_vids: Dict[str, List[int]] = {}

    for vid, spec in plan.specs.items():
        n = str(spec.name)
        name_to_vids.setdefault(n, []).append(int(vid))

        cn = _canon_name(n)
        canon_to_vids.setdefault(cn, []).append(int(vid))

    return name_to_vids, canon_to_vids


def _collect_vid_candidates_for_key(
    *,
    key: str,
    name_to_vids: Dict[str, List[int]],
    canon_to_vids: Dict[str, List[int]],
) -> List[int]:
    """
    사용자 입력 key(예: "2.W")로 plan에서 대응되는 vid 후보들을 찾는다.
    우선순위:
      1) exact match
      2) canon match (d_ prefix 제거)
      3) suffix match (plan name이 key로 끝나는 vid)
      4) canon suffix match
    """
    k = str(key)

    vids = name_to_vids.get(k)
    if vids:
        return list(vids)

    ck = _canon_name(k)
    vids = canon_to_vids.get(ck)
    if vids:
        return list(vids)

    # suffix match: plan name endswith user key
    out: List[int] = []
    for pn, vs in name_to_vids.items():
        if str(pn).endswith(k):
            out.extend(vs)
    if out:
        return out

    # canon suffix match
    out2: List[int] = []
    for pn, vs in name_to_vids.items():
        if _canon_name(pn).endswith(ck):
            out2.extend(vs)
    return out2


def _spec_kind_matches(spec, kind: str) -> bool:
    """
    bind(inputs, "input") / bind(params, "param")에 대해 plan spec.role을 이용해 우선 필터링.
    role이 없거나 예상 밖이면 완화적으로 True.
    """
    role = str(getattr(spec, "role", "") or "")
    if not role:
        return True
    if kind == "input":
        return ("input" in role) or (role == "in")
    if kind == "param":
        return ("param" in role) or ("weight" in role) or ("bias" in role)
    return True


def _choose_vid_for_tensor(
    *,
    candidates: List[int],
    plan: BindingPlan,
    t: torch.Tensor,
    kind: str,
    opts: ExecOptions,
) -> Optional[int]:
    """
    중복 name이 있을 때 올바른 vid를 선택한다.
    전략:
      1) role(kind) 매칭 후보 우선
      2) shape/dtype/device가 _assert_tensor_matches를 통과하는 첫 후보
      3) 없으면 None
    """
    if not candidates:
        return None

    # 1) kind(role) 필터
    filtered = [v for v in candidates if _spec_kind_matches(plan.specs[int(v)], kind)]
    if not filtered:
        filtered = list(candidates)

    # 2) shape/dtype/device 검증 통과하는 후보 선택
    for v in filtered:
        spec = plan.specs[int(v)]
        try:
            _assert_tensor_matches(spec, t, opts)
            return int(v)
        except Exception:
            pass

    return None


class PlannedExecutor:
    """
    Stage4+: plan 기반 실행기
      - env는 (static alloc) + (user inputs) + (user params)로만 구성
      - lowered를 순서대로 backend.op_call_out로 실행
      - in-place(out==in) op도 lowered대로 그대로 수행

    NOTE:
      - 여기서는 IRExecutor처럼 SSA rebind를 할 필요가 없음.
        lowered에서 vid는 'storage'로 해석되고, plan이 storage를 고정하기 때문.
      - 다만, outputs가 다른 텐서를 가리키는 경우(env[vid]=out)을 허용하면 확장에 유리함.
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

        # 2) robust name maps from plan (duplicate-name safe)
        name_to_vids, canon_to_vids = _build_name_maps(self.plan)

        def bind(src: Dict[str, Any], kind: str):
            for name, obj in src.items():
                t = _unwrap(obj)
                if not isinstance(t, torch.Tensor):
                    raise TypeError(f"[plan] {kind} '{name}' must be torch.Tensor-like, got {type(obj)}")

                cands = _collect_vid_candidates_for_key(
                    key=str(name),
                    name_to_vids=name_to_vids,
                    canon_to_vids=canon_to_vids,
                )
                vid = _choose_vid_for_tensor(
                    candidates=cands,
                    plan=self.plan,
                    t=t,
                    kind=kind,
                    opts=opts,
                )
                if vid is None:
                    # 보여주기용으로 known list는 짧게
                    known = list(name_to_vids.keys())
                    head = known[:24]
                    tail_note = "" if len(known) <= 24 else f" ...(+{len(known)-24})"
                    raise KeyError(
                        f"[plan] unknown or ambiguous {kind} name='{name}'. "
                        f"candidates={cands}. known={head}{tail_note}"
                    )

                spec = self.plan.specs[int(vid)]
                # 최종 검증(에러 메시지 정확히)
                _assert_tensor_matches(spec, t, opts)
                self.env[int(vid)] = t

        bind(inputs, "input")
        bind(params, "param")

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
        try:
            bk = self.backend or get_backend()
        except AssertionError as e:
            raise AssertionError(
                "Backend not set. core_v2 executor needs a backend.\n"
                "Fix: call aicf_fw.backend.set_backend(AICFBackend()) before running PlannedExecutor."
            ) from e

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
