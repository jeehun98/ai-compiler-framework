#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::layernorm_impl {

// f32: one block per row
__global__ void layernorm_fwd_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,  // nullable
    const float* __restrict__ beta,   // nullable
    float* __restrict__ y,
    float* __restrict__ mean,         // [M]
    float* __restrict__ rstd,         // [M]
    int M, int N,
    float eps);

// f16: input/output half, mean/rstd float
__global__ void layernorm_fwd_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma, // nullable
    const __half* __restrict__ beta,  // nullable
    __half* __restrict__ y,
    float* __restrict__ mean,         // [M]
    float* __restrict__ rstd,         // [M]
    int M, int N,
    float eps);

} // namespace aicf::cuda::layernorm_impl
