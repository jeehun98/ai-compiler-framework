# examples/python/aicf_fw/utils/nvtx.py
from contextlib import contextmanager
import torch

@contextmanager
def nvtx_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
