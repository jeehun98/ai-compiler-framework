#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::reduce_sum_impl {

__global__ void reduce_sum_lastdim_f32_kernel(
    const float* __restrict__ dY,
    float* __restrict__ dB,
    int M, int N);

// F16 input -> F32 output
__global__ void reduce_sum_lastdim_f16_kernel(
    const __half* __restrict__ dY,
    float* __restrict__ dB,
    int M, int N);

// half2 path: N even, dY 4B aligned
// N2 = N/2, dB is still float[N]
__global__ void reduce_sum_lastdim_f16x2_kernel(
    const __half2* __restrict__ dY,
    float* __restrict__ dB,
    int M, int N2);

} // namespace
