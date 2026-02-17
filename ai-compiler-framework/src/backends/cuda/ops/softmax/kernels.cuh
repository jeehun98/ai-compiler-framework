#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::softmax_impl {

// Softmax over last dimension.
// Interpret X as [rows, cols] where cols = last_dim, rows = numel/cols.
//
// f32: X->Y
__global__ void softmax_lastdim_f32_kernel(const float* __restrict__ x,
                                          float* __restrict__ y,
                                          int64_t rows,
                                          int64_t cols);

// f16: X->Y (accumulate in f32)
__global__ void softmax_lastdim_f16_kernel(const __half* __restrict__ x,
                                          __half* __restrict__ y,
                                          int64_t rows,
                                          int64_t cols);

} // namespace aicf::cuda::softmax_impl
