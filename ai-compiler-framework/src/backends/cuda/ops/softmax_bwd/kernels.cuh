#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::softmax_bwd_impl {

// dX = Y * (dY - sum(dY*Y))  over last dim
__global__ void softmax_bwd_lastdim_f32_kernel(const float* __restrict__ y,
                                              const float* __restrict__ dy,
                                              float* __restrict__ dx,
                                              int64_t rows,
                                              int64_t cols);

__global__ void softmax_bwd_lastdim_f16_kernel(const __half* __restrict__ y,
                                              const __half* __restrict__ dy,
                                              __half* __restrict__ dx,
                                              int64_t rows,
                                              int64_t cols);

} // namespace aicf::cuda::softmax_bwd_impl
