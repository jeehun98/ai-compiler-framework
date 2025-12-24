#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace aicf::cuda::mse_grad_impl {

__global__ void mse_grad_f32_kernel(const float* __restrict__ pred,
                                   const float* __restrict__ target,
                                   float* __restrict__ dPred,
                                   int64_t numel,
                                   float scale) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= numel) return;
  dPred[i] = (pred[i] - target[i]) * scale;
}

} // namespace aicf::cuda::mse_grad_impl
