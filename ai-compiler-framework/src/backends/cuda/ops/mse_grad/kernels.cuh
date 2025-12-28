#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::mse_grad_impl {

// grid-stride loop kernels

__global__ void mse_grad_f32_kernel(const float* __restrict__ pred,
                                    const float* __restrict__ target,
                                    float* __restrict__ dPred,
                                    int64_t numel,
                                    float scale);

__global__ void mse_grad_f16_kernel(const __half* __restrict__ pred,
                                    const __half* __restrict__ target,
                                    __half* __restrict__ dPred,
                                    int64_t numel,
                                    float scale);

// half2 vectorized (numel2 = numel/2)
__global__ void mse_grad_f16x2_kernel(const __half2* __restrict__ pred,
                                      const __half2* __restrict__ target,
                                      __half2* __restrict__ dPred,
                                      int64_t numel2,
                                      float scale);

} // namespace aicf::cuda::mse_grad_impl
