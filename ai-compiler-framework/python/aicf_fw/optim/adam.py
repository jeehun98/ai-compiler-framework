# aicf_fw/optim/adam.py
from __future__ import annotations

import torch

from aicf_fw.optim.base import Optimizer
from aicf_fw.fw.naming import opt_m_name, opt_v_name, BC1_NAME, BC2_NAME

# NEW: emit 기반 compile에서 meta/state를 EmitCtx에 등록하기 위함
try:
    from aicf_fw.fw.emit_ctx import EmitCtx
except Exception:  # optional import
    EmitCtx = object  # type: ignore


class Adam(Optimizer):
    """
    Adam optimizer (pointer-stable meta + m/v state)

    핵심 설계:
    - bc1_inv, bc2_inv는 shape=() scalar 텐서로 device에 고정 생성 (pointer stable)
    - m/v는 파라미터 텐서와 동일 shape로 device에 고정 생성 (pointer stable)
    - update_meta()는 값만 fill_로 갱신 (포인터 안정)
    - named_state_tensors()는 executor 바인딩용 이름->텐서 제공

    emit 기반 compile을 위해:
    - bind_state_to_ctx(ctx): EmitCtx에 (bc1/bc2, m/v)를 role="meta"로 등록
      -> plan이 meta/state를 'static/meta buffer'로 잡게 만들 수 있음
    """

    def __init__(
        self,
        model,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float32,
    ):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.dtype = dtype

        named_params = list(model.named_parameters())
        if len(named_params) == 0:
            raise RuntimeError("Adam: model has no parameters")

        dev = named_params[0][1].device

        # host-managed meta counter (python int)
        self.step_host: int = 0

        # device meta (pointer-stable)
        self.bc1_inv = torch.ones((), device=dev, dtype=dtype)
        self.bc2_inv = torch.ones((), device=dev, dtype=dtype)

        # m/v state (pointer-stable)
        self.m: dict[str, torch.Tensor] = {}
        self.v: dict[str, torch.Tensor] = {}
        for pname, p in named_params:
            self.m[pname] = torch.zeros_like(p)
            self.v[pname] = torch.zeros_like(p)

    # ----------------------------
    # runtime meta update
    # ----------------------------
    def update_meta(self):
        self.step_host += 1
        bc1 = 1.0 / (1.0 - (self.beta1 ** self.step_host))
        bc2 = 1.0 / (1.0 - (self.beta2 ** self.step_host))
        self.bc1_inv.fill_(float(bc1))
        self.bc2_inv.fill_(float(bc2))

    # ----------------------------
    # state tensor binding (executor input)
    # ----------------------------
    def named_state_tensors(self) -> dict[str, torch.Tensor]:
        """
        executor에 바인딩할 state 텐서들.
        key 네이밍은 기존 fw.naming 규칙을 따른다.
        """
        d: dict[str, torch.Tensor] = {
            BC1_NAME: self.bc1_inv,
            BC2_NAME: self.bc2_inv,
        }
        for pname in self.m:
            d[opt_m_name(pname)] = self.m[pname]
            d[opt_v_name(pname)] = self.v[pname]
        return d

    # alias (편의)
    def state_tensors(self) -> dict[str, torch.Tensor]:
        return self.named_state_tensors()

    # ----------------------------
    # NEW: EmitCtx에 meta/state 등록 (compile-time)
    # ----------------------------
    def bind_state_to_ctx(self, ctx: EmitCtx) -> None:
        """
        emit 기반 compile에서 optimizer state를 ctx에 value로 등록한다.
        - role은 'meta'로 둔다. (plan에서 meta를 별도로 분리하고 싶으면)
          지금 plan이 meta를 따로 처리 안하면 'static'으로 바꿔도 됨.
        """
        # ctx가 제공하는 API에 맞춰 호출. (너가 준 EmitCtx에 meta role은 존재함)
        role = "meta"

        # scalar meta
        ctx.meta_vid(BC1_NAME, shape=tuple(self.bc1_inv.shape), role=role)  # shape=()
        ctx.meta_vid(BC2_NAME, shape=tuple(self.bc2_inv.shape), role=role)

        # m/v per parameter
        for pname, mt in self.m.items():
            ctx.meta_vid(opt_m_name(pname), shape=tuple(mt.shape), role=role)
        for pname, vt in self.v.items():
            ctx.meta_vid(opt_v_name(pname), shape=tuple(vt.shape), role=role)

    # ----------------------------
    # NEW: compile에서 statics dict로 바로 넣고 싶을 때
    # ----------------------------
    def state_as_statics(self) -> dict[str, torch.Tensor]:
        """
        CompiledTrainStep에 statics로 합치기 쉽게 state dict만 반환.
        compile.py에서 statics.update(opt.state_as_statics()) 이렇게 쓰면 됨.
        """
        return self.named_state_tensors()
