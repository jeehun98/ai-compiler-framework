#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace aicf::cuda::ops::eltwise {

__global__ void add_f32_kernel(
    float* out,
    const float* a,
    const float* b,
    int64_t n
);

__global__ void relu_f32_kernel(
    float* out,
    const float* x,
    int64_t n
);

} // namespace
