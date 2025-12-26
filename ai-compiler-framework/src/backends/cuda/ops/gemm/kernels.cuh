// ============================================================================
// src/backends/cuda/ops/gemm/kernels.cuh   (DECLARATIONS ONLY)
// ============================================================================

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::gemm_impl {

// -------------------------
// f32 naive
// -------------------------
__global__ void gemm_f32_naive_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K);

__global__ void gemm_f32_naive_transB_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B, // stored [N,K] (strided)
                                            float* __restrict__ C,
                                            int M, int N, int K,
                                            int64_t B_stride0, int64_t B_stride1);

// -------------------------
// WMMA TensorCore (A,B half -> C half)  [OUTPUT F16]
// internal acc is float, but store is half
// -------------------------
__global__ void gemm_f16_tc_wmma_nn_out_f16_kernel(const __half* __restrict__ A,
                                                  const __half* __restrict__ B,
                                                  __half* __restrict__ C,
                                                  int M, int N, int K);

__global__ void gemm_f16_tc_wmma_tn_out_f16_kernel(const __half* __restrict__ A, // stored [K,M]
                                                  const __half* __restrict__ B, // [K,N]
                                                  __half* __restrict__ C,
                                                  int M, int N, int K);

__global__ void gemm_f16_tc_wmma_nt_out_f16_kernel(const __half* __restrict__ A, // [M,K]
                                                  const __half* __restrict__ B, // stored [N,K]
                                                  __half* __restrict__ C,
                                                  int M, int N, int K);

} // namespace aicf::cuda::gemm_impl
