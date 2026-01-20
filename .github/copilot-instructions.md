# AI Compiler Framework — Copilot Instructions

## Project Identity
This is a **CUDA-centric AI compiler + runtime framework**, not a full ML training framework. The core mission: transform dynamic AI model logic into static, deterministic GPU execution graphs that can be captured and replayed.

**Key constraint**: Most costs are paid at compile-time, not at runtime. This differs fundamentally from typical ML workflows (model → train → infer). Here: IR → compile → capture → replay.

## Architecture Layers (Critical to Understand)

### 1. **Intermediate Representation (IR)** — *What* to do
- Located: `examples/python/aicf_fw/core_v2/ir.py`
- Models computations abstractly: operation type, data flow, logical execution order
- Does NOT encode: CUDA grids/blocks, shared memory, Tensor Core usage
- **Philosophy**: IR carries only "meaning", not "method"

### 2. **Planner** — Binding strategy
- Located: `examples/python/aicf_fw/core_v2/plan.py`
- Analyzes IR values and classifies them as:
  - **input**: external injection (x, t)
  - **param**: model parameters only (regex: `^\d+\.(W|b)$`)
  - **static**: runtime-allocated (intermediates, optimizer states)
- Stage 3/6 rules: optimizer states (`opt.*`), gradients (`d_*`, `grad*`), backward values are forced static
- See `BindingPlan`, `PlanOptions` for core concepts

### 3. **Runtime** — How to execute deterministically
- Located: `examples/python/aicf_fw/core_v2/exec.py`
- Responsibilities: Graph definition, execution strategy (CapturePlan), GraphExec lifecycle
- **Does NOT**: implement kernels, micro-optimize, branch on hardware
- Key pattern: IR → Graph → (CUDA Graph Capture) → GraphExec
- `PlannedExecutor` manages tensor binding and shape/dtype/device validation

### 4. **Backend (CUDA)** — Where it actually runs
- Located: `src/backends/cuda/ops/`, `src/backends/cuda/registry/`
- Kernel implementations for: matmul, reduce_sum, relu, bias_add, adam_step, batchnorm, etc.
- Registry pattern: each op variant declares tile size, data type, and optimization (e.g., Tensor Core usage)
- See `src/backends/cuda/registry/register_all.cpp` for all registered kernel variants

## Critical Data Flows

### Tracing & IR Construction
1. Wrap computation in `@tracing(ir)` context (see [trace.py](examples/python/aicf_fw/core_v2/trace.py))
2. Use `SymTensor` (metadata only, no actual computation) to represent values
3. Call ops like `linear()`, `relu()`, `adam_step()` which emit `IRNode`s into the graph
4. Identity-based caching: torch tensors mapped by `data_ptr()`, Python objects by `id()`
5. **Key pattern in trace.py**: 
   - `obj_cache` (Dict[int, IRValue]): maps Python object identity → IR value
   - `torch_cache` (Dict[Tuple, IRValue]): maps tensor ptr + shape/dtype/device → IR value

### Compilation Pipeline
```
User model (PyTorch Module)
    ↓
trace_ir(step_fn) — creates IR with SymTensors
    ↓
build_binding_plan(ir) — classifies inputs/params/statics via regex + naming rules
    ↓
GraphExec(ir, plan) — runtime binder, manages CUDA graph lifecycle
    ↓
executor.execute(inputs={}, params={}) — CUDA Graph capture → replay
```

### Validator Pattern
- `_assert_tensor_matches()` in [exec.py](examples/python/aicf_fw/core_v2/exec.py) validates shape, dtype, device before execution
- Guards against caller shape/type mismatches via `ExecOptions.check_*` flags
- Windows-specific: `_bootstrap_aicf_cuda()` handles DLL path setup for `aicf_cuda._C` module

## Project-Specific Patterns

### SymTensor vs torch.Tensor
- **SymTensor**: used during tracing, carries only metadata (shape, dtype, device, name)
- **torch.Tensor**: real data, used at runtime binding in `PlannedExecutor`
- Strict separation: core_v2 framework code never allocates real tensors; that's the executor's job

### Naming Conventions Matter
- Model parameters: `{layer_id}.{W|b}` (e.g., "0.W", "2.b")
- Optimizer state: `opt.*` prefix → always static allocation
- Gradients: `d_*`, `grad*`, `dY` → always static allocation
- Inputs: "x", "t" (configurable in `PlanOptions.input_names`)
- See `fw/naming.py` for stage-specific naming utilities

### Build System
- **Tool**: CMake (Ninja generator preferred for faster builds)
- **Requires**: CUDA Toolkit, Python 3.12+, PyTorch
- **Key build flags**:
  - `-DAICF_ENABLE_NVTX=ON` for profiling instrumentation (default: ON)
  - `-DCMAKE_CUDA_ARCHITECTURES=86` (or your GPU's compute capability; 80 for A100, 89 for RTX 4090)
  - `-DCMAKE_BUILD_TYPE=Release` for optimized kernels
  - `-DPython_ROOT_DIR` and `-DTorch_DIR` on Windows (see [build_command.md](build_command.md))
- **Output**: `build/python/aicf_cuda/_C.*.so` (Linux) or `_C.*.pyd` (Windows)
- **Full build command**: 
  ```bash
  cd build
  cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DAICF_ENABLE_NVTX=ON -DCMAKE_CUDA_ARCHITECTURES=86
  ninja
  ```

### Testing Pattern
- **Probe tests**: [examples/python/python_binding_test/](examples/python/python_binding_test/) verify individual ops (e.g., `v2_gemm_probe.py`)
- **Framework tests**: [examples/python/python_framework_test/](examples/python/python_framework_test/) test end-to-end training steps
- **Environment**: Use `AICF_WARMUP=<N>` to control CUDA Graph warmup iterations (default: 2)
  - Higher warmup for more complex graphs or when debugging kernel registration

## Integration Points

### Python ↔ C++ Bridge (Critical for Windows)
- CUDA kernels accessed via `aicf_cuda._C` module (loaded in [exec.py](examples/python/aicf_fw/core_v2/exec.py))
- **Bootstrap pattern** (`_bootstrap_aicf_cuda()`):
  1. Adds `build/python/` to `sys.path`
  2. On Windows: calls `os.add_dll_directory()` for DLL resolution
  3. Imports compiled module: `from aicf_cuda import _C`
- Kernel invocation: `GraphExec` calls backend dispatch based on IR op names + dtype
- **Troubleshooting**: If `aicf_cuda._C` fails to import, check:
  - Build completed successfully (`build/python/aicf_cuda/_C.pyd` exists)
  - Python path includes `build/python/`
  - DLL dependencies available (CUDA Runtime, cuBLAS, etc.)

### Module Architecture (fw package)
- **core_v2/**: IR, planning, tracing (pure Python, no CUDA calls)
  - `ir.py`: `IRValue`, `IRNode`, `IRGraph`
  - `trace.py`: `@tracing()` context, caching logic
  - `plan.py`: `BindingPlan`, role classification rules
  - `exec.py`: `GraphExec`, tensor binding, execution
- **backend/**: OP kind enums, ABI definitions, validation
  - `abi.py`, `validate.py`: shared C++/Python contracts
- **fw/**: High-level framework (model wrapper, optimizer binding, naming utilities)
- **nn/**: Neural network layers with `forward_ir()` (IR generation) + `forward_torch()` (inference)

## Common Commands

```bash
# Build (from ai-compiler-framework directory)
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DAICF_ENABLE_NVTX=ON -DCMAKE_CUDA_ARCHITECTURES=86
ninja

# Run a probe test (validates single op)
python examples/python/python_binding_test/v2_linear_probe.py

# Run framework test (end-to-end)
python examples/python/python_framework_test/v2_fw_mvp_train_step_test.py

# Set warmup iterations
AICF_WARMUP=5 python examples/python/python_binding_test/v2_linear_probe.py
```

## Anti-Patterns to Avoid

1. **Do NOT**: call real torch ops inside a `tracing()` context expecting them to be recorded as IR nodes; use `SymTensor` + explicit `ops.*` calls instead
2. **Do NOT**: manually allocate tensors in the planner; `PlannedExecutor` owns memory via `allocate_static_env()`
3. **Do NOT**: assume backend kernel selection is automatic; check `register_all.cpp` for dtype/layout support
4. **Do NOT**: mix ROLE_PARAM and ROLE_STATIC; `PlanOptions` rules are strict (optimizer state must be static)
5. **Do NOT**: call `compile_train_step()` without valid model parameters; framework expects at least one named parameter

## When Adding New Operations

1. **Define IR op type**: Add to IR (name convention: snake_case)
2. **Add C++ enum**: `OpKind::YourOp` in backend headers
3. **Implement kernel**: `src/backends/cuda/ops/your_op/launcher.cu`
4. **Register**: Add to `register_all.cpp` with priority & compute capability
5. **Test**: Create probe in `examples/python/python_binding_test/v2_your_op_probe.py`

## Files to Start With

- [ai-compiler-framework/docs/architecture/overview.md](../../docs/architecture/overview.md) — layered architecture rationale
- [ai-compiler-framework/docs/architecture/runtime.md](../../docs/architecture/runtime.md) — runtime lifecycle & CUDA Graph capture strategy
- [examples/python/aicf_fw/core_v2/ir.py](examples/python/aicf_fw/core_v2/ir.py) — IRValue, IRNode, IRGraph definitions
- [examples/python/aicf_fw/core_v2/exec.py](examples/python/aicf_fw/core_v2/exec.py) — PlannedExecutor, tensor binding logic
- [examples/python/aicf_fw/fw/compile.py](examples/python/aicf_fw/fw/compile.py) — end-to-end compilation pipeline
- [src/backends/cuda/registry/register_all.cpp](src/backends/cuda/registry/register_all.cpp) — all available kernel variants
