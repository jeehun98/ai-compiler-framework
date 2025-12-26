#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace aicf::cuda::bias_add_impl {

__global__ void bias_add_f32_kernel(const float* __restrict__ Y,
                                   const float* __restrict__ bias,
                                   float* __restrict__ Out,
                                   int M, int N);

__global__ void bias_add_f16_kernel(const __half* __restrict__ Y,
                                   const __half* __restrict__ bias,
                                   __half* __restrict__ Out,
                                   int M, int N);

__global__ void bias_add_f16x2_kernel(const __half2* __restrict__ Y,
                                      const __half2* __restrict__ bias,
                                      __half2* __restrict__ Out,
                                      int M, int N2); // N2 = N/2

} // namespace aicf::cuda::bias_add_impl
