#pragma once
#include <cuda_runtime.h>

namespace aicf::cuda::gemm_impl {

// naive: C[M,N] = A[M,K] * B[K,N], row-major
__global__ void gemm_f32_naive_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int N, int K);

} // namespace aicf::cuda::gemm_impl
