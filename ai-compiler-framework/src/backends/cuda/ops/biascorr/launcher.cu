#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

#include <aicf/core/status.hpp>

#include <aicf/backends/cuda/ops/biascorr/api.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>

#include "aicf/backends/cuda/ops/_common/shim/status.hpp"

namespace aicf::cuda {

// ---- small attr helpers ----
static inline bool attr_get_f32(const void* attr, const char* key, float& out) {
  if (!attr) return false;
  const AttrPack* p = (const AttrPack*)attr;
  if (!p->items || p->size <= 0) return false;
  for (int i = 0; i < p->size; ++i) {
    const AttrKV& kv = p->items[i];
    if (!kv.key) continue;
    // exact match
    const char* a = kv.key;
    const char* b = key;
    while (*a && *b && *a == *b) { ++a; ++b; }
    if (*a == 0 && *b == 0) {
      if (kv.val.tag != AttrTag::kF32) return false;
      out = kv.val.f32;
      return true;
    }
  }
  return false;
}

__global__ void biascorr_kernel(const int32_t* step,
                                float beta1, float beta2,
                                float* bc1_inv, float* bc2_inv) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int t = step[0];
    if (t < 1) t = 1;  // safety

    // bc_inv = 1 / (1 - beta^t)
    // use powf on device (fine for scalar)
    float b1t = powf(beta1, (float)t);
    float b2t = powf(beta2, (float)t);

    float d1 = 1.0f - b1t;
    float d2 = 1.0f - b2t;

    // avoid div0 if beta==1 (shouldn't happen)
    if (fabsf(d1) < 1e-20f) d1 = 1e-20f;
    if (fabsf(d2) < 1e-20f) d2 = 1e-20f;

    bc1_inv[0] = 1.0f / d1;
    bc2_inv[0] = 1.0f / d2;
  }
}

aicf::Status biascorr_v0(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;
  if (num_inputs != 1 || num_outputs != 2) return aicf::Status::InvalidArgument;

  const TensorDesc& S = inputs[0];
  TensorDesc& O1 = outputs[0];
  TensorDesc& O2 = outputs[1];

  // step: int32 scalar
  if (S.dtype != DType::kI32) return aicf::Status::NotImplemented;
  if (!S.contiguous) return aicf::Status::NotImplemented;

  // outputs: f32 scalars
  if (O1.dtype != DType::kF32 || O2.dtype != DType::kF32) return aicf::Status::NotImplemented;
  if (!O1.contiguous || !O2.contiguous) return aicf::Status::NotImplemented;

  // scalar shape check (rank=0 or rank=1 len=1 허용)
  auto is_scalar = [](const TensorDesc& d) -> bool {
    if (d.r.rank == 0) return true;
    if (d.r.rank == 1 && d.shape[0] == 1) return true;
    return false;
  };
  if (!is_scalar(S) || !is_scalar(O1) || !is_scalar(O2)) return aicf::Status::NotImplemented;

  float beta1 = 0.0f, beta2 = 0.0f;
  if (!attr_get_f32(attr, "beta1", beta1)) return aicf::Status::InvalidArgument;
  if (!attr_get_f32(attr, "beta2", beta2)) return aicf::Status::InvalidArgument;

  biascorr_kernel<<<1, 32, 0, stream>>>(
      (const int32_t*)S.data,
      beta1, beta2,
      (float*)O1.data, (float*)O2.data);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// KernelVariant glue
// -------------------------
static size_t ws_biascorr(const TensorDesc*, int, const void*) { return 0; }

static bool supported_biascorr(
    const TensorDesc* in, int ni,
    const TensorDesc* out, int no,
    const void* attr) {

  if (!in || !out) return false;
  if (ni != 1 || no != 2) return false;
  if (in[0].dtype != DType::kI32) return false;
  if (!in[0].contiguous) return false;

  if (out[0].dtype != DType::kF32 || out[1].dtype != DType::kF32) return false;
  if (!out[0].contiguous || !out[1].contiguous) return false;

  auto is_scalar = [](const TensorDesc& d) -> bool {
    if (d.r.rank == 0) return true;
    if (d.r.rank == 1 && d.shape[0] == 1) return true;
    return false;
  };
  if (!is_scalar(in[0]) || !is_scalar(out[0]) || !is_scalar(out[1])) return false;

  // require attrs beta1/beta2
  float tmp = 0.f;
  if (!attr_get_f32(attr, "beta1", tmp)) return false;
  if (!attr_get_f32(attr, "beta2", tmp)) return false;

  return true;
}

static aicf::Status launch_biascorr(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {

  return biascorr_v0(inputs, num_inputs, outputs, num_outputs, attr, workspace, workspace_bytes, stream);
}

KernelVariant make_biascorr_variant() {
  KernelVariant kv{};
  kv.name = "biascorr_f32_v0";
  kv.priority = 0;
  kv.query_workspace = ws_biascorr;
  kv.supported = supported_biascorr;
  kv.launch = launch_biascorr;
  return kv;
}

} // namespace aicf::cuda
