#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace aicf::cuda {

template <typename T>
__global__ void copy_1d_kernel(const T* __restrict__ x, T* __restrict__ y, int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i];
}

} // namespace aicf::cuda
