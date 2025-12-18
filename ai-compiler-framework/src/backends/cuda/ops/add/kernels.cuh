#pragma once
#include <cuda_runtime.h>

namespace aicf::cuda::add_impl {

__global__ void add_f32_kernel(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ out,
                              int N);

} // namespace aicf::cuda::add_impl
