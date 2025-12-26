#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::sgd_step_impl {

__global__ void sgd_step_f32_kernel(float* __restrict__ param,
                                   const float* __restrict__ grad,
                                   int64_t numel,
                                   float lr);

__global__ void sgd_step_f16_kernel(__half* __restrict__ param,
                                   const __half* __restrict__ grad,
                                   int64_t numel,
                                   float lr);

__global__ void sgd_step_f16_half2_kernel(__half2* __restrict__ param2,
                                         const __half2* __restrict__ grad2,
                                         int64_t numel2,
                                         float lr);

} // namespace aicf::cuda::sgd_step_impl
