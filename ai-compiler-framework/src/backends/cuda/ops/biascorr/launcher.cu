#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cmath>

#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

// -------------------------
// cuda error -> Status (core-free)
// -------------------------
static inline Status cuda_to_status(cudaError_t e) {
  return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}
static inline Status cuda_last_status() {
  return cuda_to_status(cudaGetLastError());
}

// -------------------------
// AttrBlob schema: BiasCorr
// payload: float32 beta1, float32 beta2  (little-endian)
// schema_id: 'BCOR' = 0x524F4342
// schema_id==0 allowed? -> 여기서는 "필수"로 둠(더 단순)
// -------------------------
static constexpr uint32_t kSchema_BCOR = 0x524F4342u;

static inline float read_f32_le(const uint8_t* p) {
  float v;
  std::memcpy(&v, p, sizeof(float));
  return v;
}

static inline bool get_betas_from_attr(const void* attr, float* beta1, float* beta2) {
  if (!attr || !beta1 || !beta2) return false;
  const AttrBlob& a = *static_cast<const AttrBlob*>(attr);
  if (a.schema_id != kSchema_BCOR) return false;
  if (!a.data || a.bytes < 8) return false;
  const uint8_t* p = static_cast<const uint8_t*>(a.data);
  *beta1 = read_f32_le(p + 0);
  *beta2 = read_f32_le(p + 4);
  // NaN 방지 (필수키 체크 느낌)
  if (!(*beta1 == *beta1) || !(*beta2 == *beta2)) return false;
  return true;
}

// -------------------------
// helpers
// -------------------------
static inline bool is_scalar0d_or_1elem1d(const TensorDesc& d) {
  if (d.rank() == 0) return true;
  if (d.rank() == 1 && d.shape[0] == 1) return true;
  return false;
}

static inline bool is_i32_scalar_contig(const TensorDesc& d) {
  if (d.dtype != DType::kI32) return false;
  if (!is_scalar0d_or_1elem1d(d)) return false;
  // rank==0일 때 contiguous가 false로 들어올 수 있으면 완화해도 되는데,
  // 지금은 SgdStep/StepInc 쪽과 다르게 "단순 contig" 유지.
  return d.contiguous;
}

static inline bool is_f32_scalar_contig(const TensorDesc& d) {
  if (d.dtype != DType::kF32) return false;
  if (!is_scalar0d_or_1elem1d(d)) return false;
  return d.contiguous;
}

static size_t biascorr_workspace(const TensorDesc*, int, const void*) { return 0; }

// -------------------------
// kernel (definition)
// -------------------------
namespace biascorr_impl {

__global__ void biascorr_kernel(const int32_t* step,
                               float beta1, float beta2,
                               float* bc1_inv, float* bc2_inv) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int t = step[0];
    if (t < 1) t = 1;

    float b1t = powf(beta1, (float)t);
    float b2t = powf(beta2, (float)t);

    float d1 = 1.0f - b1t;
    float d2 = 1.0f - b2t;

    if (fabsf(d1) < 1e-20f) d1 = 1e-20f;
    if (fabsf(d2) < 1e-20f) d2 = 1e-20f;

    bc1_inv[0] = 1.0f / d1;
    bc2_inv[0] = 1.0f / d2;
  }
}

} // namespace biascorr_impl

// -------------------------
// Variant glue
// Contract:
// inputs[0] = step (i32 scalar)
// outputs[0] = bc1_inv (f32 scalar)
// outputs[1] = bc2_inv (f32 scalar)
// attr: AttrBlob schema 'BCOR' with (beta1,beta2)
// -------------------------
static inline bool biascorr_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 2) return false;

  const TensorDesc& S  = inputs[0];
  const TensorDesc& O1 = outputs[0];
  const TensorDesc& O2 = outputs[1];

  if (!is_i32_scalar_contig(S)) return false;
  if (!is_f32_scalar_contig(O1) || !is_f32_scalar_contig(O2)) return false;

  float b1 = 0.f, b2 = 0.f;
  if (!get_betas_from_attr(attr, &b1, &b2)) return false;
  return true;
}

static bool biascorr_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {
  return biascorr_check(inputs, num_inputs, outputs, num_outputs, attr);
}

static Status biascorr_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!biascorr_check(inputs, num_inputs, outputs, num_outputs, attr)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& S  = inputs[0];
  TensorDesc& O1 = outputs[0];
  TensorDesc& O2 = outputs[1];

  float beta1 = 0.f, beta2 = 0.f;
  (void)get_betas_from_attr(attr, &beta1, &beta2);

  cudaGetLastError(); // clear
  biascorr_impl::biascorr_kernel<<<1, 32, 0, stream>>>(
      (const int32_t*)S.data,
      beta1, beta2,
      (float*)O1.data,
      (float*)O2.data);

  return cuda_last_status();
}

// IMPORTANT: register_all.cpp가 찾는 심볼
KernelVariant make_biascorr_variant() {
  KernelVariant v{};
  v.name = "biascorr_f32_v0";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = kSchema_BCOR;
  v.launch = biascorr_launch;
  v.supported = biascorr_supported;
  v.query_workspace = biascorr_workspace;
  return v;
}

} // namespace aicf::cuda
