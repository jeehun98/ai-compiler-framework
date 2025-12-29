#pragma once
#include <cuda_runtime.h>
#include <math.h>

namespace aicf::cuda {

__global__ void adam_step_f32_kernel(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    int64_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bc1_inv,
    float bc2_inv) {

  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  float g  = grad[i];
  float mi = m[i] = beta1 * m[i] + (1.0f - beta1) * g;
  float vi = v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

  float mhat = mi * bc1_inv;
  float vhat = vi * bc2_inv;

  param[i] -= lr * mhat / (sqrtf(vhat) + eps);
}

} // namespace aicf::cuda
