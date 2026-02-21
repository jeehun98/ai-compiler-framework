#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace aicf::cuda::xent_impl {

// small utils
__global__ void xent_set_int_scalar(int* p, int v);

__global__ void xent_finalize_loss_f32(
    const float* __restrict__ loss_sum,
    const int* __restrict__ valid,
    int reduction,
    float* __restrict__ out_loss);

__global__ void xent_count_valid_i32(
    const int32_t* __restrict__ t,
    int N,
    int ignore_index,
    int* __restrict__ out_valid);

// Forward
__global__ void xent_fwd_sum_f32(
    const float* __restrict__ logits,
    const int32_t* __restrict__ targets,
    int N, int C,
    int ignore_index,
    float* __restrict__ out_loss_sum,
    int* __restrict__ out_valid);

// Backward
__global__ void xent_bwd_f32(
    const float* __restrict__ logits,
    const int32_t* __restrict__ targets,
    int N, int C,
    int ignore_index,
    const float* __restrict__ grad_loss,
    const int* __restrict__ valid,
    int reduction,
    float* __restrict__ dlogits);

} // namespace aicf::cuda::xent_impl