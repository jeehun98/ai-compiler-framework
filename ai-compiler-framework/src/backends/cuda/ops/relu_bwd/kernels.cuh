#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace aicf::cuda::relu_bwd_impl {

__global__ void relu_bwd_f32_kernel(const float* __restrict__ Y,
                                   const float* __restrict__ dOut,
                                   float* __restrict__ dY,
                                   int64_t numel) {
  int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (i >= numel) return;
  const float y = Y[i];
  dY[i] = (y > 0.0f) ? dOut[i] : 0.0f;
}

} // namespace aicf::cuda::relu_bwd_impl
