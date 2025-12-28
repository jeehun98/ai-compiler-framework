#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/mse_grad/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>

// common shim
#include "aicf/backends/cuda/ops/_common/shim/launch.hpp"
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"
#include "aicf/backends/cuda/ops/_common/shim/validate.hpp"

#include "kernels.cuh"

#include <string_view>

namespace aicf::cuda {

// -------------------------
// kernels
// -------------------------
namespace mse_grad_impl {

__global__ void mse_grad_f32_kernel(const float* __restrict__ pred,
                                    const float* __restrict__ target,
                                    float* __restrict__ dPred,
                                    int64_t numel,
                                    float scale) {
  // grid-stride loop
  for (int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
       i < numel;
       i += (int64_t)blockDim.x * (int64_t)gridDim.x) {
    dPred[i] = (pred[i] - target[i]) * scale;
  }
}

__global__ void mse_grad_f16_kernel(const __half* __restrict__ pred,
                                    const __half* __restrict__ target,
                                    __half* __restrict__ dPred,
                                    int64_t numel,
                                    float scale) {
  // grid-stride loop (scalar)
  for (int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
       i < numel;
       i += (int64_t)blockDim.x * (int64_t)gridDim.x) {
    const float p = __half2float(pred[i]);
    const float t = __half2float(target[i]);
    dPred[i] = __float2half_rn((p - t) * scale);
  }
}

__global__ void mse_grad_f16x2_kernel(const __half2* __restrict__ pred,
                                      const __half2* __restrict__ target,
                                      __half2* __restrict__ dPred,
                                      int64_t numel2,
                                      float scale) {
  // half2 arithmetic path + grid-stride loop
  const __half2 s2 = __float2half2_rn(scale);
  for (int64_t i = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
       i < numel2;
       i += (int64_t)blockDim.x * (int64_t)gridDim.x) {
    const __half2 p2 = pred[i];
    const __half2 t2 = target[i];
    // (p2 - t2) * s2
    dPred[i] = __hmul2(__hsub2(p2, t2), s2);
  }
}

} // namespace mse_grad_impl

// -------------------------
// Attr helpers (binding v0.2 compatible)
// -------------------------
static inline bool attr_get_f32(const void* attr, const char* key, float* out_val) {
  if (!attr || !out_val) return false;
  const auto* pack = static_cast<const aicf::cuda::AttrPack*>(attr);
  if (!pack->items || pack->size <= 0) return false;

  const std::string_view k(key);
  for (int i = 0; i < pack->size; ++i) {
    const auto& kv = pack->items[i];
    if (!kv.key) continue;
    if (std::string_view(kv.key) == k) {
      if (kv.val.tag == aicf::cuda::AttrTag::kF32) {
        *out_val = kv.val.f32;
        return true;
      }
      return false;
    }
  }
  return false;
}

// -------------------------
// helpers
// -------------------------
static inline bool is_f32_contig(const TensorDesc& T) {
  return (T.dtype == DType::kF32) && T.contiguous;
}
static inline bool is_f16_contig(const TensorDesc& T) {
  return (T.dtype == DType::kF16) && T.contiguous;
}

static inline bool same_shape(const TensorDesc& A, const TensorDesc& B) {
  if (A.rank() != B.rank()) return false;
  for (int64_t i = 0; i < A.rank(); ++i) {
    if (A.shape[i] != B.shape[i]) return false;
  }
  return true;
}

static inline bool compute_numel(const TensorDesc& T, int64_t* out_numel) {
  if (!out_numel) return false;
  const int64_t r = T.rank();
  if (r < 1) return false;

  int64_t n = 1;
  for (int64_t i = 0; i < r; ++i) {
    const int64_t d = T.shape[i];
    if (d <= 0) return false;
    n *= d;
  }
  if (n <= 0) return false;
  *out_numel = n;
  return true;
}

static inline bool mse_grad_check_dt(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    bool (*is_ok)(const TensorDesc&)) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& P = inputs[0];
  const TensorDesc& T = inputs[1];
  const TensorDesc& G = outputs[0];

  if (!is_ok(P) || !is_ok(T) || !is_ok(G)) return false;
  if (P.rank() < 1) return false;
  if (!same_shape(P, T)) return false;
  if (!same_shape(P, G)) return false;

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return false;
  return true;
}

static inline bool mse_grad_vec2_ok(const TensorDesc& P, const TensorDesc& T, const TensorDesc& G) {
  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return false;
  if ((numel & 1) != 0) return false;

  constexpr size_t kAlign = 4;
  if (!aicf::cuda::shim::is_aligned_data(P, kAlign)) return false;
  if (!aicf::cuda::shim::is_aligned_data(T, kAlign)) return false;
  if (!aicf::cuda::shim::is_aligned_data(G, kAlign)) return false;
  return true;
}

static size_t mse_grad_workspace(const TensorDesc*, int, const void*) { return 0; }

// -------------------------
// Launch config helper
// -------------------------
static inline int compute_blocks_1d(int64_t n, int threads) {
  // choose blocks based on problem size, but do NOT clamp to 65535
  // because kernel uses grid-stride loop and can safely run with fewer blocks.
  int64_t blocks64 = (n + threads - 1) / threads;
  if (blocks64 < 1) blocks64 = 1;

  // avoid ridiculous blocks count; cap to something sane for occupancy
  // (this is a performance cap, not correctness cap)
  const int64_t kMaxBlocks = 4096;
  if (blocks64 > kMaxBlocks) blocks64 = kMaxBlocks;

  return (int)blocks64;
}

// -------------------------
// Variant: F32
// -------------------------
static bool mse_grad_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return mse_grad_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f32_contig);
}

static aicf::Status mse_grad_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!mse_grad_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f32_contig)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& T = inputs[1];
  TensorDesc& G = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return aicf::Status::InvalidArgument;

  float scale = 0.0f;
  const bool has_scale = attr_get_f32(attr, "scale", &scale);
  if (!has_scale) scale = 2.0f / (float)numel;

  constexpr int kThreads = 256;
  const int blocks = compute_blocks_1d(numel, kThreads);

  mse_grad_impl::mse_grad_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)P.data, (const float*)T.data, (float*)G.data, numel, scale);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_mse_grad_f32_variant() {
  KernelVariant v{};
  v.name = "mse_grad_f32";
  v.priority = 0;
  v.flags = 0;
  v.launch = mse_grad_f32_launch;
  v.supported = mse_grad_f32_supported;
  v.query_workspace = mse_grad_workspace;
  return v;
}

// -------------------------
// Variant: F16
// -------------------------
static bool mse_grad_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return mse_grad_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f16_contig);
}

static aicf::Status mse_grad_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!mse_grad_check_dt(inputs, num_inputs, outputs, num_outputs, &is_f16_contig)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& T = inputs[1];
  TensorDesc& G = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return aicf::Status::InvalidArgument;

  float scale = 0.0f;
  const bool has_scale = attr_get_f32(attr, "scale", &scale);
  if (!has_scale) scale = 2.0f / (float)numel;

  constexpr int kThreads = 256;

  // half2 fastpath (correctness-safe due to vec2_ok checks)
  if (mse_grad_vec2_ok(P, T, G)) {
    const int64_t numel2 = numel / 2;
    const int blocks = compute_blocks_1d(numel2, kThreads);

    mse_grad_impl::mse_grad_f16x2_kernel<<<blocks, kThreads, 0, stream>>>(
        (const __half2*)P.data, (const __half2*)T.data, (__half2*)G.data, numel2, scale);
  } else {
    const int blocks = compute_blocks_1d(numel, kThreads);

    mse_grad_impl::mse_grad_f16_kernel<<<blocks, kThreads, 0, stream>>>(
        (const __half*)P.data, (const __half*)T.data, (__half*)G.data, numel, scale);
  }

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_mse_grad_f16_variant() {
  KernelVariant v{};
  v.name = "mse_grad_f16";
  v.priority = 0;
  v.flags = 0;
  v.launch = mse_grad_f16_launch;
  v.supported = mse_grad_f16_supported;
  v.query_workspace = mse_grad_workspace;
  return v;
}

} // namespace aicf::cuda
