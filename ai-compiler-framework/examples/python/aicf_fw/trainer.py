from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Dict, List

import torch

from .backend import get_backend
from .tensor import Tensor
from .modules.base import Parameter  # (남겨둬도 됨; 이제 의존하지 않음)
from . import autograd as AG


# -----------------------------
# helpers
# -----------------------------
def _to_torch(x: Any) -> torch.Tensor:
    """
    Accept:
      - torch.Tensor
      - aicf_fw.tensor.Tensor (wrapper) having attribute .t which is torch.Tensor
      - any object with attribute .t that is torch.Tensor
    """
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, Tensor):
        return x.t
    if hasattr(x, "t") and isinstance(getattr(x, "t"), torch.Tensor):
        return getattr(x, "t")
    raise TypeError(f"expected torch.Tensor or Tensor wrapper, got {type(x)}")


def _contig(x: Any) -> torch.Tensor:
    """
    Always return contiguous torch.Tensor.
    Handles both wrapper Tensor and torch.Tensor.
    """
    x_t = _to_torch(x)
    return x_t if x_t.is_contiguous() else x_t.contiguous()


def _require_cuda(x: Any, what: str) -> None:
    x_t = _to_torch(x)
    if not x_t.is_cuda:
        raise RuntimeError(f"{what}: CUDA tensor required")


def _same_shape(a: Any, b: Any, what: str) -> None:
    a_t = _to_torch(a)
    b_t = _to_torch(b)
    if tuple(a_t.shape) != tuple(b_t.shape):
        raise RuntimeError(f"{what}: shape mismatch {tuple(a_t.shape)} vs {tuple(b_t.shape)}")


# -----------------------------
# config
# -----------------------------
@dataclass
class TrainerConfig:
    lr: float = 1e-3
    mode: str = "eager"     # "eager" | "bench" | "capture"
    log_every: int = 10


# -----------------------------
# trainer
# -----------------------------
class Trainer:
    def __init__(self, model: Any, optim: Any = None, cfg: Optional[TrainerConfig] = None) -> None:
        self.model = model
        self.optim = optim
        self.cfg = cfg or TrainerConfig()

        self.backend = get_backend()
        self._cache: Dict[str, torch.Tensor] = {}

    # -----------------------------
    # parameter discovery (robust)
    # -----------------------------
    def _is_param_like(self, obj: Any) -> bool:
        """
        Parameter-like object:
          - has attribute .data
          - obj.data can be converted to torch.Tensor via _to_torch
        This avoids relying on a single Parameter class, which can differ across modules/bindings.
        """
        if obj is None or not hasattr(obj, "data"):
            return False
        try:
            _ = _to_torch(getattr(obj, "data"))
            return True
        except Exception:
            return False

    def _parameters(self) -> Iterable[Any]:
        """
        Robust parameter collection:
          1) prefer model.parameters() if it exists and returns usable param-like objects
          2) otherwise recursively walk object graph and collect param-like objects
        """
        # 1) prefer explicit parameters()
        if hasattr(self.model, "parameters") and callable(self.model.parameters):
            try:
                ps = list(self.model.parameters())
                ps2 = [p for p in ps if self._is_param_like(p)]
                if len(ps2) > 0:
                    return ps2
            except Exception:
                pass

        found: List[Any] = []
        visited: set[int] = set()

        def walk(obj: Any):
            if obj is None:
                return
            oid = id(obj)
            if oid in visited:
                return
            visited.add(oid)

            if self._is_param_like(obj):
                found.append(obj)
                return

            # common containers
            if isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
                return
            if isinstance(obj, (list, tuple, set)):
                for v in obj:
                    walk(v)
                return

            # generic objects
            if hasattr(obj, "__dict__"):
                for v in obj.__dict__.values():
                    walk(v)

        walk(self.model)
        return found

    # -----------------------------
    # backward starting from dy
    # -----------------------------
    def _backward_from_dy(self, dy: Any) -> None:
        """
        dy: gradient wrt model output
        This function:
          - reads recorded ops from AG.tape()
          - propagates gradients backward
          - writes p.grad for parameters
          - applies SGD update (torch fallback)
        """
        g = _contig(dy)
        _require_cuda(g, "dy")

        ops = AG.tape().ops

        for op in reversed(ops):
            if op.kind == "relu":
                x = _contig(op.x)
                _require_cuda(x, "relu.x")

                # keep dtype consistent if relu_bwd expects same dtype
                if g.dtype != x.dtype:
                    x = x.to(dtype=g.dtype)

                g = AG.aicf_relu_bwd(g, x)
                g = _contig(g)

            elif op.kind == "linear":
                x = _contig(op.x)
                w = _contig(op.w_param.data)
                _require_cuda(x, "linear.x")
                _require_cuda(w, "linear.w")

                # ---- force fp32 backward GEMM (for stability + kernel availability) ----
                g32 = g.float() if g.dtype != torch.float32 else g
                x32 = x.float() if x.dtype != torch.float32 else x
                w32 = w.float() if w.dtype != torch.float32 else w

                B = g32.shape[0]
                Dout = g32.shape[1]
                Din = x32.shape[1]

                # ---- dx = g @ w ----
                dx = torch.empty((B, Din), device=g32.device, dtype=torch.float32).contiguous()
                AG._C.op_call(
                    AG._C.OpKind.Gemm,
                    [g32.contiguous(), w32.contiguous()],
                    [dx],
                    {"transA": False, "transB": False},
                )

                # ---- dW = g^T @ x ----
                gT = g32.t().contiguous()
                dW = torch.empty((Dout, Din), device=g32.device, dtype=torch.float32).contiguous()
                AG._C.op_call(
                    AG._C.OpKind.Gemm,
                    [gT, x32.contiguous()],
                    [dW],
                    {"transA": False, "transB": False},
                )

                # ---- db = sum(g, axis=0) ----
                if getattr(op, "b_param", None) is not None and op.b_param is not None:
                    db = g32.sum(dim=0).contiguous()
                    op.b_param.grad = db

                op.w_param.grad = dW
                g = dx

            else:
                raise RuntimeError(f"Unknown op kind: {op.kind}")

        # apply update
        self._sgd_step()

    # -----------------------------
    # optimizer (safe torch fallback)
    # -----------------------------
    def _sgd_step(self) -> None:
        params = list(self._parameters())

        # debug print (keep until stable)
        print("num params:", len(params))
        for i, p in enumerate(params[:8]):
            w = _to_torch(getattr(p, "data"))
            g_obj = getattr(p, "grad", None)
            if g_obj is None:
                print(f"p{i}: {type(p)} w {tuple(w.shape)} {w.dtype} grad None")
            else:
                g = _to_torch(g_obj)
                print(f"p{i}: {type(p)} w {tuple(w.shape)} {w.dtype} grad {tuple(g.shape)} {g.dtype}")

        # lr policy
        lr = float(self.cfg.lr)
        if self.optim is not None and hasattr(self.optim, "lr"):
            try:
                lr = float(self.optim.lr)
            except Exception:
                pass

        # update
        for p in params:
            w = _to_torch(getattr(p, "data"))
            g_obj = getattr(p, "grad", None)
            if g_obj is None:
                continue
            g = _to_torch(g_obj)

            # match weight dtype
            if w.dtype != g.dtype:
                g = g.to(dtype=w.dtype)

            w.add_(g, alpha=-lr)

            # clear grad
            try:
                p.grad = None
            except Exception:
                pass

    # -----------------------------
    # train step (eager)
    # -----------------------------
    def train_step_eager(self, batch) -> torch.Tensor:
        """
        End-to-end eager step:
          1) clear tape
          2) forward
          3) loss (torch)
          4) dy (aicf mse_grad)
          5) backward from dy using tape
        """
        AG.tape().clear()

        x, t = batch
        x_t = _to_torch(x)
        t_t = _to_torch(t)
        _require_cuda(x_t, "batch.x")
        _require_cuda(t_t, "batch.t")

        # forward (model expects Tensor wrapper; keep as-is)
        y = self.model(Tensor(x_t) if not isinstance(x, Tensor) else x)
        y_t = _to_torch(y)

        # loss (torch fallback ok)
        loss_t = torch.mean((y_t - t_t.to(dtype=y_t.dtype)) ** 2)

        # dy (aicf op)
        dy = AG.aicf_mse_grad(y_t, t_t)
        dy_t = _to_torch(dy)

        # backward + update
        self._backward_from_dy(dy_t)

        return loss_t.detach()

    # -----------------------------
    # fit loop
    # -----------------------------
    def fit(self, dl: Iterable, steps: int = 100) -> None:
        it = iter(dl)
        for i in range(steps):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl)
                batch = next(it)

            loss = self.train_step_eager(batch)

            if self.cfg.log_every > 0 and (i % self.cfg.log_every) == 0:
                print(f"step={i} loss={loss.item():.6f}")
