#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace aicf::cuda::ops::eltwise {

void add_f32(
    float* out,
    const float* a,
    const float* b,
    int64_t numel,
    cudaStream_t stream
);

void relu_f32(
    float* out,
    const float* x,
    int64_t numel,
    cudaStream_t stream
);

} // namespace aicf::cuda::ops::eltwise
