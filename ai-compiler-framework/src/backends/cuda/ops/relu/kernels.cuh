#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace aicf::cuda::relu_impl {

__global__ void relu_f32_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               int N);

__global__ void relu_f16_kernel(const __half* __restrict__ in,
                               __half* __restrict__ out,
                               int N);

__global__ void relu_f16x2_kernel(const __half2* __restrict__ in,
                                 __half2* __restrict__ out,
                                 int N2); // N2 = N/2

} // namespace aicf::cuda::relu_impl
