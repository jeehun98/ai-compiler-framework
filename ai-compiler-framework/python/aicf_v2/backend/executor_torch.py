# aicf_v2/backend/executor_torch.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from aicf_v2.fw.emit_ctx import Program, TensorSpec, BufferId


def _to_torch_dtype(dt: str) -> torch.dtype:
    if dt in ("f16", "float16"):
        return torch.float16
    if dt in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dt in ("f32", "float32"):
        return torch.float32
    raise ValueError(f"unsupported dtype: {dt}")


def _alloc(spec: TensorSpec) -> torch.Tensor:
    device = torch.device(spec.device)
    dtype = _to_torch_dtype(spec.dtype)
    return torch.empty(spec.shape, device=device, dtype=dtype)


@dataclass
class TorchExecutor:
    """
    - prog.lowered_ops를 그대로 순회하며 torch 연산으로 실행
    - provided: buffer_id -> torch.Tensor 주입 (입력/파라미터)
    """
    debug: bool = False

    def run(
        self,
        prog: Program,
        provided: Dict[BufferId, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[BufferId, torch.Tensor]]:
        # 1) allocate all buffers (v0: 1 buffer = 1 tensor)
        bufs: Dict[BufferId, torch.Tensor] = {}
        for bid, spec in enumerate(prog.buffer_specs):
            bufs[bid] = _alloc(spec)

        # 2) override with provided buffers
        for bid, t in provided.items():
            bufs[bid] = t

        # 3) execute lowered ops
        for i, op in enumerate(prog.lowered_ops):
            k = op["kernel"]
            args = op["args"]
            meta = op.get("meta", {})

            if self.debug:
                print(f"[torch-exec] #{i} {k} args={args} meta={meta}")

            if k.startswith("gemm"):
                # args: (A, W, Y)
                a = bufs[args[0]]
                w = bufs[args[1]]
                y = bufs[args[2]]

                trans_w = bool(meta.get("trans_w", meta.get("transB", True)))
                # 우리 Linear는 W가 [N,K]고 trans_w=True면 y = a @ w.T
                out = a @ (w.t() if trans_w else w)

                # write into y (copy) to emulate "destination buffer"
                y.copy_(out)

            elif k.startswith("bias_add"):
                # args: (X, B, Y)
                x = bufs[args[0]]
                b = bufs[args[1]]
                y = bufs[args[2]]

                axis = int(meta.get("broadcast_axis", -1))
                inplace = bool(meta.get("inplace", False))

                # 최소: axis=-1 케이스만 지원 (지금 Linear가 이거만 씀)
                if axis not in (-1, x.dim() - 1):
                    raise NotImplementedError(f"bias_add only supports last axis for now, got axis={axis}")

                # b: [N] broadcast -> [*, N]
                out = x + b

                if inplace:
                    # y가 x랑 같은 버퍼일 수도 있으니 y로 write
                    y.copy_(out)
                else:
                    y.copy_(out)

            else:
                raise NotImplementedError(f"unknown kernel: {k}")

        # 기본 output: 마지막 op의 마지막 arg를 out0로
        last = prog.lowered_ops[-1]
        out0 = bufs[last["args"][-1]]
        return {"out0": out0}, bufs
