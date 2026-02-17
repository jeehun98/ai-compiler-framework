#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace aicf::cuda::xent_impl {

// small utils
__global__ void xent_set_int_scalar(int* p, int v);

// Forward: accumulate (sum loss, valid count)
__global__ void xent_fwd_sum_f32(
    const float* __restrict__ logits,      // [N,C]
    const int32_t* __restrict__ targets,   // [N]
    int N, int C,
    int ignore_index,
    float* __restrict__ out_loss_sum,      // scalar
    int* __restrict__ out_valid);          // scalar

// Forward finalize: (sum or mean) -> out_loss[0]
__global__ void xent_finalize_loss_f32(
    const float* __restrict__ loss_sum,    // scalar
    const int* __restrict__ valid,         // scalar
    int reduction,                         // 0 mean, 1 sum
    float* __restrict__ out_loss);         // scalar

// Backward helper: count valid targets (ignore_index 제외)
__global__ void xent_count_valid_i32(
    const int32_t* __restrict__ t,
    int N,
    int ignore_index,
    int* __restrict__ out_valid);          // scalar

// Backward: dlogits
__global__ void xent_bwd_f32(
    const float* __restrict__ logits,      // [N,C]
    const int32_t* __restrict__ targets,   // [N]
    int N, int C,
    int ignore_index,
    const float* __restrict__ grad_loss,   // device scalar
    const int* __restrict__ valid,         // device scalar (mean이면 valid_count, sum이면 1)
    int reduction,                         // 0 mean, 1 sum
    float* __restrict__ dlogits);          // [N,C]

} // namespace aicf::cuda::xent_impl
