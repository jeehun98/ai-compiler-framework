#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::copy_impl {

template <typename T>
__global__ void copy_1d_kernel(const T* __restrict__ x,
                               T* __restrict__ y,
                               int64_t n) {
  // grid-stride (sane + no 65535 issue)
  for (int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
       i < n;
       i += (int64_t)blockDim.x * (int64_t)gridDim.x) {
    y[i] = x[i];
  }
}

} // namespace aicf::cuda::copy_impl
