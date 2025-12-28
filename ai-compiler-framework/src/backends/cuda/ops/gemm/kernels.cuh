// ============================================================================
// src/backends/cuda/ops/gemm/kernels.cuh   (DECLARATIONS ONLY)
// ============================================================================

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::gemm_impl {

// -------------------------
// f32 naive: unified STRIDED (covers NN/TN/NT/TT via MatView2D)
// -------------------------
__global__ void gemm_f32_naive_strided_kernel(
    const float* __restrict__ A, int64_t Ars, int64_t Acs,
    const float* __restrict__ B, int64_t Brs, int64_t Bcs,
    float* __restrict__ C, int64_t Crs, int64_t Ccs,
    int M, int N, int K);

// -------------------------
// WMMA TensorCore: unified STRIDED (A,B half -> C half)
// - Covers NN/TN/NT/TT via MatView2D (stride-swapped logical views)
// - GLOBAL->SMEM packing handles WMMA layout (A row-major, B col-major)
// - acc float, store half
// -------------------------
__global__ void gemm_f16_tc_wmma_out_f16_strided_kernel(
    const __half* __restrict__ A, int64_t Ars, int64_t Acs, int64_t Am, int64_t Ak,
    const __half* __restrict__ B, int64_t Brs, int64_t Bcs, int64_t Bk, int64_t Bn,
    __half* __restrict__ C, int64_t Crs, int64_t Ccs, int64_t Cm, int64_t Cn);

} // namespace aicf::cuda::gemm_impl
