#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>   // sqrtf

namespace aicf::cuda {

// ------------------------------
// Adam update (f32) v1 (in-place / out is Mout,Vout but reads from Mout,Vout)
//  - kept for compatibility
// ------------------------------
__global__ void adam_step_f32_kernel_v1(
    float* __restrict__ P,
    const float* __restrict__ G,
    float* __restrict__ M,
    float* __restrict__ V,
    int64_t n,
    float lr, float beta1, float beta2, float eps,
    const float* __restrict__ bc1_inv_ptr,
    const float* __restrict__ bc2_inv_ptr) {

  __shared__ float s_bc1_inv;
  __shared__ float s_bc2_inv;
  if (threadIdx.x == 0) {
    s_bc1_inv = *bc1_inv_ptr;
    s_bc2_inv = *bc2_inv_ptr;
  }
  __syncthreads();

  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float g = G[i];
  float m = M[i];
  float v = V[i];

  m = beta1 * m + (1.0f - beta1) * g;
  v = beta2 * v + (1.0f - beta2) * (g * g);

  const float m_hat = m * s_bc1_inv;
  const float v_hat = v * s_bc2_inv;

  const float denom = sqrtf(v_hat) + eps;
  P[i] = P[i] - lr * (m_hat / denom);

  M[i] = m;
  V[i] = v;
}

// ------------------------------
// Adam update (f32) v2 (true out-of-place support)
//  - Reads previous state from Min/Vin
//  - Writes updated state to Mout/Vout
//  - P is updated in-place via Pout pointer
// ------------------------------
__global__ void adam_step_f32_kernel_v2(
    float* __restrict__ Pout,
    const float* __restrict__ G,
    const float* __restrict__ Min,
    const float* __restrict__ Vin,
    float* __restrict__ Mout,
    float* __restrict__ Vout,
    int64_t n,
    float lr, float beta1, float beta2, float eps,
    const float* __restrict__ bc1_inv_ptr,
    const float* __restrict__ bc2_inv_ptr) {

  __shared__ float s_bc1_inv;
  __shared__ float s_bc2_inv;
  if (threadIdx.x == 0) {
    s_bc1_inv = *bc1_inv_ptr;   // 1/(1-beta1^t)
    s_bc2_inv = *bc2_inv_ptr;   // 1/(1-beta2^t)
  }
  __syncthreads();

  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const float g = G[i];
  float m = Min[i];
  float v = Vin[i];

  m = beta1 * m + (1.0f - beta1) * g;
  v = beta2 * v + (1.0f - beta2) * (g * g);

  const float m_hat = m * s_bc1_inv;
  const float v_hat = v * s_bc2_inv;

  const float denom = sqrtf(v_hat) + eps;
  Pout[i] = Pout[i] - lr * (m_hat / denom);

  Mout[i] = m;
  Vout[i] = v;
}

} // namespace aicf::cuda
