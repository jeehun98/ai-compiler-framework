#pragma once
#include <cuda_runtime.h>

namespace aicf::cuda::relu_impl {

__global__ void relu_f32_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               int N);

} // namespace aicf::cuda::relu_impl
