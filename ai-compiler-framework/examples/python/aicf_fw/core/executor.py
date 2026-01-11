# aicf_fw/core/executor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from aicf_fw.backend import get_backend
from aicf_fw.core.ir import IRGraph


@dataclass
class IRExecutor:
    """
    IR-only executor:
      - Executes lowered ops list (artifact.lowered)
      - Uses env: vid(int) -> torch.Tensor
      - Dispatch via backend.op_call_out(op, inputs_t, outputs_t, attrs)
    """
    ir: IRGraph
    lowered: List[Dict[str, Any]]
    env: Dict[int, torch.Tensor]
    backend: Any

    @staticmethod
    def from_artifact(art: Any) -> "IRExecutor":
        if not hasattr(art, "runtime_env"):
            raise RuntimeError("IRExecutor.from_artifact: artifact has no runtime_env()")

        env = art.runtime_env()
        if not isinstance(env, dict) or len(env) == 0:
            aname = getattr(art, "name", "<unnamed>")
            raise RuntimeError(
                f"IRExecutor.from_artifact: empty env for artifact={aname}. "
                "Attach env first: art.attach_env(vid->torch.Tensor)."
            )
        return IRExecutor(ir=art.ir, lowered=art.lowered, env=dict(env), backend=art.backend)

    def _get_t(self, vid: int, *, op: str = "?", io: str = "?") -> torch.Tensor:
        t = self.env.get(int(vid), None)
        if t is None:
            v = self.ir.values.get(int(vid), None)
            meta = ""
            if v is not None:
                meta = f" name={v.name} shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
            raise RuntimeError(f"IRExecutor: missing runtime tensor for vid={vid} ({io} of {op}).{meta}")
        return t

    @torch.no_grad()
    def run(self, *, debug_nan: bool = False) -> None:
        bk = self.backend or get_backend()

        def _t_meta(t: torch.Tensor) -> str:
            return f"shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}"

        def _nonfinite_count(t: torch.Tensor) -> tuple[int, int]:
            return int(torch.isnan(t).sum().item()), int(torch.isinf(t).sum().item())

        for idx, item in enumerate(self.lowered):
            op = item["op"]
            attrs = dict(item.get("attrs", {}))
            in_vids = list(item.get("inputs", []))
            out_vids = list(item.get("outputs", []))

            inputs_t = [self._get_t(v, op=op, io="in") for v in in_vids]
            outputs_t = [self._get_t(v, op=op, io="out") for v in out_vids]

            # -----------------------------
            # CRITICAL: force in-place semantics for adam_step
            # inputs:  [p_in, g_in, m_in, v_in, bc1, bc2]
            # outputs: [p_out, m_out, v_out]
            # -----------------------------
            if op == "adam_step":
                if len(inputs_t) < 4 or len(outputs_t) < 3:
                    raise RuntimeError(f"IRExecutor: malformed adam_step io. inputs={len(inputs_t)} outputs={len(outputs_t)}")
                outputs_t = [inputs_t[0], inputs_t[2], inputs_t[3]]

            # dispatch
            bk.op_call_out(op, inputs_t, outputs_t, attrs)

            # SSA rebind: bind output vids to the tensor handles we used
            for ov, ot in zip(out_vids, outputs_t):
                self.env[int(ov)] = ot

            if debug_nan and len(outputs_t) > 0:
                for j, (ov, ot) in enumerate(zip(out_vids, outputs_t)):
                    nn, ni = _nonfinite_count(ot)
                    if nn or ni:
                        ins_meta = ", ".join([f"{iv}:{_t_meta(it)}" for iv, it in zip(in_vids, inputs_t)])
                        outs_meta = ", ".join([f"{ov}:{_t_meta(ot)}" for ov, ot in zip(out_vids, outputs_t)])
                        raise RuntimeError(
                            f"IRExecutor: first non-finite after op#{idx:02d} '{op}' out#{j} vid={int(ov)} "
                            f"(nan={nn}, inf={ni}).\n"
                            f"  inputs: {ins_meta}\n"
                            f"  outputs: {outs_meta}\n"
                            f"  attrs: {attrs}"
                        )
