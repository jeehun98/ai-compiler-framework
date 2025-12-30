// ============================================================================
// src/backends/cuda/ops/layernorm/kernels.cuh  (DECLARATIONS ONLY)
// ============================================================================

#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::layernorm_impl {

// -------------------- forward --------------------
__global__ void layernorm_fwd_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int M, int N,
    float eps);

__global__ void layernorm_fwd_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    __half* __restrict__ y,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int M, int N,
    float eps);

// -------------------- backward: dx --------------------
__global__ void layernorm_bwd_dx_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const float* __restrict__ gamma, // may be nullptr => gamma=1
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ dx,
    int M, int N);

__global__ void layernorm_bwd_dx_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ dy,
    const __half* __restrict__ gamma, // may be nullptr
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    __half* __restrict__ dx,
    int M, int N);

// -------------------- backward: dgamma/dbeta (f32 outputs) --------------------
__global__ void layernorm_bwd_dg_db_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int M, int N);

__global__ void layernorm_bwd_dg_db_f16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ dy,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    int M, int N);

} // namespace aicf::cuda::layernorm_impl
