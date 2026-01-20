#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace aicf::cuda::reduce_sum_impl {

// dY: [M, N] flattened row-major contiguous (M = product(leading dims), N = last dim)
// out: [N]

// F32 -> F32
__global__ void reduce_sum_rows_f32_to_f32_kernel(
    const float* __restrict__ dY,
    float* __restrict__ out,
    int M, int N);

// F16 -> F32
__global__ void reduce_sum_rows_f16_to_f32_kernel(
    const __half* __restrict__ dY,
    float* __restrict__ out,
    int M, int N);

// half2 path: N even, dY 4B aligned, out is float[N]
__global__ void reduce_sum_rows_f16x2_to_f32_kernel(
    const __half2* __restrict__ dY,
    float* __restrict__ out,
    int M, int N2);

// F16 -> F16
__global__ void reduce_sum_rows_f16_to_f16_kernel(
    const __half* __restrict__ dY,
    __half* __restrict__ out,
    int M, int N);

// half2 path: N even, dY/out 4B aligned, out is half[N]
__global__ void reduce_sum_rows_f16x2_to_f16_kernel(
    const __half2* __restrict__ dY,
    __half2* __restrict__ out2,
    int M, int N2);

} // namespace aicf::cuda::reduce_sum_impl
