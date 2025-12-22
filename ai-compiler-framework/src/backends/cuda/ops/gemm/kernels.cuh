#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>   // ✅ __half

namespace aicf::cuda::gemm_impl {

// naive f32
__global__ void gemm_f32_naive_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K);

// ✅ NEW: TC GEMM (half inputs, float output), supports NN/TN/NT via flags
__global__ void gemm_f16_tc_wmma_kernel(const __half* __restrict__ A,
                                       const __half* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K,
                                       bool transA,
                                       bool transB);

} // namespace aicf::cuda::gemm_impl
