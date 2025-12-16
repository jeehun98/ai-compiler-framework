# AI Compiler Framework

A CUDA-centric AI compiler framework that lowers high-level model graphs
into optimized, capture-safe GPU execution plans.

## Goals
- Static graph construction from dynamic model logic
- Deterministic CUDA Graph capture & replay
- Kernel-level specialization (Tensor Core, warp specialization)
- Clear separation: IR → Planner → Runtime → Backend

## Non-Goals
- Not a full training framework
- Not a PyTorch/XLA replacement
- No dynamic autograd at runtime

## Status
🚧 Early development. APIs are unstable.
