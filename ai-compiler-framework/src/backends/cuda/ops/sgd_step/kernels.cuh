#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::sgd_step_impl {

// out-of-place: param_out = param_in - lr * grad
__global__ void sgd_step_f32_kernel(const float* __restrict__ param_in,
                                   const float* __restrict__ grad,
                                   float* __restrict__ param_out,
                                   int64_t numel,
                                   float lr);

__global__ void sgd_step_f16_kernel(const __half* __restrict__ param_in,
                                   const __half* __restrict__ grad,
                                   __half* __restrict__ param_out,
                                   int64_t numel,
                                   float lr);

__global__ void sgd_step_f16_half2_kernel(const __half2* __restrict__ param_in2,
                                         const __half2* __restrict__ grad2,
                                         __half2* __restrict__ param_out2,
                                         int64_t numel2,
                                         float lr);

} // namespace aicf::cuda::sgd_step_impl
