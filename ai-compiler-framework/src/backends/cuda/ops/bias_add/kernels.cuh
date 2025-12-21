#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace aicf::cuda::bias_add_impl {

__global__ void bias_add_f32_kernel(const float* __restrict__ Y,
                                   const float* __restrict__ bias,
                                   float* __restrict__ Out,
                                   int M, int N) {
  const int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  const int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= M || col >= N) return;

  const int idx = row * N + col;
  Out[idx] = Y[idx] + bias[col];
}

} // namespace aicf::cuda::bias_add_impl
