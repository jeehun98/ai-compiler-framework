#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace aicf::cuda::sgd_step_impl {

__global__ void sgd_step_f32_kernel(float* __restrict__ param,
                                   const float* __restrict__ grad,
                                   int64_t numel,
                                   float lr) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= numel) return;
  param[i] -= lr * grad[i];
}

} // namespace aicf::cuda::sgd_step_impl
