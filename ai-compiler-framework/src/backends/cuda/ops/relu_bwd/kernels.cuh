#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::relu_bwd_impl {

__global__ void relu_bwd_f32_kernel(const float* __restrict__ Y,
                                   const float* __restrict__ dOut,
                                   float* __restrict__ dY,
                                   int64_t numel);

__global__ void relu_bwd_f16_kernel(const __half* __restrict__ Y,
                                   const __half* __restrict__ dOut,
                                   __half* __restrict__ dY,
                                   int64_t numel);

__global__ void relu_bwd_f16x2_kernel(const __half2* __restrict__ Y,
                                     const __half2* __restrict__ dOut,
                                     __half2* __restrict__ dY,
                                     int64_t numel2); // numel2 = numel/2

} // namespace aicf::cuda::relu_bwd_impl
