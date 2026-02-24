#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace aicf::cuda::gemm_epilogue_impl {

// F32 Forward
__global__ void gemm_bias_relu_f32_kernel(
    const float* __restrict__ A, int64_t Ars, int64_t Acs,
    const float* __restrict__ B, int64_t Brs, int64_t Bcs,
    const float* __restrict__ Bias, int64_t Bs,
    float* __restrict__ C, int64_t Crs, int64_t Ccs,
    int M, int N, int K, int relu_enable);

// F32 Backward: dBias with optional ReLU mask (mask from Y)
__global__ void bwd_bias_relu_mask_f32_kernel(
    const float* __restrict__ dY,
    float* __restrict__ dBias,
    const float* __restrict__ Y,
    int M, int N,
    int64_t Yrs, int64_t Ycs,   // stride for Y (and dY assumed same layout)
    int relu_enable);

// F16 TC Forward (WMMA)
__global__ void gemm_f16_tc_wmma_bias_relu_kernel(
    const __half* __restrict__ A, int64_t Ars, int64_t Acs, int64_t Am, int64_t Ak,
    const __half* __restrict__ B, int64_t Brs, int64_t Bcs, int64_t Bk, int64_t Bn,
    const __half* __restrict__ Bias, int64_t Bs,
    __half* __restrict__ C, int64_t Crs, int64_t Ccs, int64_t Cm, int64_t Cn,
    int relu_enable);

} // namespace aicf::cuda::gemm_epilogue_impl