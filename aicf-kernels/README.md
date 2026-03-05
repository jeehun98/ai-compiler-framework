# aicf-kernels — CUDA Kernel Sandbox

Lightweight CUDA kernel experimentation environment (build-isolated).

## Structure
- `common/` : shared utilities (error check, timer, validation)
- `src/` : individual kernels (one `.cu` = one executable)
- `scripts/` : profiling helpers (ncu / nsys)
- `out/` : profiling outputs

## Build (Windows + Ninja)
> Run in **x64 Native Tools Command Prompt for VS 2022** (or Developer PowerShell for VS)

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

Run:
```bat
build\bin\vector_add.exe --n 16777216 --iters 200
```

## Profiling

### Nsight Compute (ncu)
Fast pass (recommended for iteration):
```bat
scripts\ncu_fast.bat build\bin\vector_add.exe --n 16777216 --iters 10
```

Full pass (heavy, use when you want a final report):
```bat
scripts\ncu_full.bat build\bin\vector_add.exe --n 16777216 --iters 10
```

### Nsight Systems (nsys)
```bat
scripts\nsys.bat build\bin\vector_add.exe --n 16777216 --iters 200
```

Artifacts:
- `out/ncu/*.ncu-rep`
- `out/nsys/*.nsys-rep`
