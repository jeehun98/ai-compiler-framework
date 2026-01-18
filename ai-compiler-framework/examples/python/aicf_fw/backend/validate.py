from __future__ import annotations
import torch

def ensure_cuda(t: torch.Tensor, name: str) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor")
    if not t.is_cuda:
        raise ValueError(f"{name} must be CUDA tensor")

def ensure_contig(t: torch.Tensor, name: str) -> None:
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
