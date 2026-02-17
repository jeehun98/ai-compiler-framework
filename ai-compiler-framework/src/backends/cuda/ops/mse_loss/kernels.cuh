#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::mse_loss_impl {

// out[0] += sum((pred - target)^2)  (atomic accumulation)
// then launcher optionally scales for mean.
__global__ void mse_loss_sum_f32_kernel(const float* __restrict__ pred,
                                       const float* __restrict__ target,
                                       float* __restrict__ out,   // 1 elem (f32)
                                       int64_t numel);

__global__ void mse_loss_sum_f16_kernel(const __half* __restrict__ pred,
                                       const __half* __restrict__ target,
                                       float* __restrict__ out,   // 1 elem (f32)
                                       int64_t numel);

// out[0] *= scale (single element)
__global__ void scale_f32_scalar_kernel(float* __restrict__ out, float scale);

} // namespace aicf::cuda::mse_loss_impl
