// ============================================================================
// src/backends/cuda/ops/batchnorm/kernels.cuh
// - Declarations must EXACTLY match the definitions in launcher.cu
// ============================================================================

#pragma once
#include <cuda_fp16.h>

namespace aicf::cuda::bn_impl {

// fwd training stats (atomic sum / sumsq)
__global__ void bn_fwd_stats_f16_atomic(
    const __half* __restrict__ x,
    float* __restrict__ sum,     // [C]
    float* __restrict__ sumsq,   // [C]
    int N, int C, int HW);

// fwd apply (mean/var given). If save_mean/save_rstd != nullptr, write them too.
__global__ void bn_fwd_apply_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma, // nullable
    const __half* __restrict__ beta,  // nullable
    const float* __restrict__ mean,   // [C]
    const float* __restrict__ var,    // [C]  (NOTE: for training path, launcher passes "var" scratch)
    __half* __restrict__ y,
    float* __restrict__ save_mean,    // nullable [C]
    float* __restrict__ save_rstd,    // nullable [C]
    int N, int C, int HW,
    float eps);

// bwd: per-channel sums (dy_hat, dy_hat*xhat) where dy_hat = dy*gamma (gamma may be null => 1)
__global__ void bn_bwd_sums_f16_atomic(
    const __half* __restrict__ x,
    const __half* __restrict__ dy,
    const __half* __restrict__ gamma, // nullable (gamma=1)
    const float* __restrict__ mean,   // [C]
    const float* __restrict__ rstd,   // [C]
    float* __restrict__ sum_dy,       // [C]
    float* __restrict__ sum_dy_xhat,  // [C]
    int N, int C, int HW);

// bwd: dx (uses sum buffers from bn_bwd_sums_f16_atomic)
__global__ void bn_bwd_dx_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ dy,
    const __half* __restrict__ gamma, // nullable
    const float* __restrict__ mean,   // [C]
    const float* __restrict__ rstd,   // [C]
    const float* __restrict__ sum_dy,       // [C]
    const float* __restrict__ sum_dy_xhat,  // [C]
    __half* __restrict__ dx,
    int N, int C, int HW);

// bwd: dgamma/dbeta (float outputs)  dbeta[c]=sum(dy), dgamma[c]=sum(dy*xhat)
__global__ void bn_bwd_dg_db_f16_atomic(
    const __half* __restrict__ x,
    const __half* __restrict__ dy,
    const float* __restrict__ mean,   // [C]
    const float* __restrict__ rstd,   // [C]
    float* __restrict__ dgamma,       // [C]
    float* __restrict__ dbeta,        // [C]
    int N, int C, int HW);

} // namespace aicf::cuda::bn_impl
