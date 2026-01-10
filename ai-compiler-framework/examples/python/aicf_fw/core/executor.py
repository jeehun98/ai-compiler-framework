# aicf_fw/core/executor.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from aicf_fw.backend import get_backend
from aicf_fw.core.ir import IRGraph


@dataclass
class IRExecutor:
    """
    IR-only executor:
      - Executes artifact.lowered (already lowered ops list)
      - Uses artifact.env : vid(int) -> torch.Tensor
      - Dispatch via backend.op_call_out(op, inputs_t, outputs_t, attrs)

    This is NOT a CUDA graph replay. It's an eager interpreter for lowered ops.
    """
    ir: IRGraph
    lowered: List[Dict[str, Any]]
    env: Dict[int, torch.Tensor]
    backend: Any

    # -------------------------
    # Constructors
    # -------------------------
    @staticmethod
    def from_artifact(art: Any) -> "IRExecutor":
        if not hasattr(art, "runtime_env"):
            raise RuntimeError("IRExecutor.from_artifact: artifact has no runtime_env() (did you patch artifact.py?)")
        env = art.runtime_env()
        if not isinstance(env, dict) or len(env) == 0:
            raise RuntimeError(
                "IRExecutor.from_artifact: empty env. "
                "compile_and_capture() must attach env (vid -> torch.Tensor)."
            )
        return IRExecutor(ir=art.ir, lowered=art.lowered, env=env, backend=art.backend)

    # -------------------------
    # Core helpers
    # -------------------------
    def _get_t(self, vid: int) -> torch.Tensor:
        t = self.env.get(int(vid), None)
        if t is None:
            # make it debuggable
            v = self.ir.values.get(int(vid), None)
            meta = ""
            if v is not None:
                meta = f" name={v.name} shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
            raise RuntimeError(f"IRExecutor: missing runtime tensor for vid={vid}.{meta}")
        return t

    def _ensure_out_rebind(self, out_vid: int, out_t: torch.Tensor) -> None:
        """
        For SSA outputs:
          - Some ops are in-place (bias_add writes into same buffer)
          - Some ops write into an output buffer tensor.
        We always bind out_vid to outputs_t[0] tensor handle after op_call.
        """
        self.env[int(out_vid)] = out_t

    # -------------------------
    # Run
    # -------------------------
    @torch.no_grad()
    def run(self) -> None:
        """
        Execute lowered ops sequentially.
        """
        bk = self.backend or get_backend()

        for item in self.lowered:
            op = item["op"]
            attrs = dict(item.get("attrs", {}))
            in_vids = list(item.get("inputs", []))
            out_vids = list(item.get("outputs", []))

            inputs_t = [self._get_t(v) for v in in_vids]
            outputs_t = [self._get_t(v) for v in out_vids]

            # dispatch
            bk.op_call_out(op, inputs_t, outputs_t, attrs)

            # rebind outputs (SSA)
            for ov, ot in zip(out_vids, outputs_t):
                self._ensure_out_rebind(ov, ot)
