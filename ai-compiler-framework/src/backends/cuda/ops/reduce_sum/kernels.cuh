#pragma once
#include <cuda_runtime.h>

namespace aicf::cuda::reduce_sum_impl {

// Each thread computes one column (j) reduction over M rows.
// dY is row-major [M,N], contiguous.
__global__ void reduce_sum_lastdim_f32_kernel(const float* __restrict__ dY,
                                             float* __restrict__ dB,
                                             int M, int N) {
  const int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (col >= N) return;

  float acc = 0.0f;

  // Sum over rows
  // dY[row*N + col]
  for (int row = 0; row < M; ++row) {
    acc += dY[row * N + col];
  }

  dB[col] = acc;
}

} // namespace aicf::cuda::reduce_sum_impl
