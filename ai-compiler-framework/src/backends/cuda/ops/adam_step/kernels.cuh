#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace aicf::cuda {

// Adam update (f32)
// bc1_inv_ptr / bc2_inv_ptr: device pointers to scalar floats (rank-0 tensors)
__global__ void adam_step_f32_kernel_v1(
    float* __restrict__ P,
    const float* __restrict__ G,
    float* __restrict__ M,
    float* __restrict__ V,
    int64_t n,
    float lr, float beta1, float beta2, float eps,
    const float* __restrict__ bc1_inv_ptr,
    const float* __restrict__ bc2_inv_ptr) {

  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float g = G[i];
  float m = M[i];
  float v = V[i];

  m = beta1 * m + (1.0f - beta1) * g;
  v = beta2 * v + (1.0f - beta2) * (g * g);

  // load bias-correction inverses from device scalar tensors
  const float bc1_inv = *bc1_inv_ptr;   // = 1/(1-beta1^t)
  const float bc2_inv = *bc2_inv_ptr;   // = 1/(1-beta2^t)

  const float m_hat = m * bc1_inv;
  const float v_hat = v * bc2_inv;

  const float denom = rsqrtf(v_hat + eps); // 1/sqrt(v_hat+eps)
  P[i] = P[i] - lr * (m_hat * denom);

  M[i] = m;
  V[i] = v;
}

} // namespace aicf::cuda
