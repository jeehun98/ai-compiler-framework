#include <cuda_runtime.h>

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
// public API implementation
// -------------------------
aicf::Status mse_grad_f32(const float* pred,
                          const float* target,
                          float* dPred,
                          int64_t numel,
                          float scale,
                          aicf::Stream stream) {
  if (!pred || !target || !dPred || numel <= 0) {
    return aicf::Status::InvalidArgument;
  }

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  const int block = 256;
  const int grid = (int)((numel + block - 1) / block);

  mse_grad_impl::mse_grad_f32_kernel<<<grid, block, 0, s>>>(
      pred, target, dPred, numel, scale);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

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

// -------------------------
// Contract:
//
// inputs[0] = pred   f32 contig, rank>=1
// inputs[1] = target f32 contig, same shape
// outputs[0]= dPred  f32 contig, same shape
//
// Attr:
//   scale (optional, f32):
//     if provided -> use it
//     else -> default scale = 2.0f / numel  (MSE mean gradient)
// -------------------------
static inline bool mse_grad_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& P = inputs[0];
  const TensorDesc& T = inputs[1];
  const TensorDesc& G = outputs[0];

  if (!is_f32_contig(P) || !is_f32_contig(T) || !is_f32_contig(G)) return false;
  if (P.rank() < 1) return false;
  if (!same_shape(P, T)) return false;
  if (!same_shape(P, G)) return false;

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return false;

  return true;
}

static bool mse_grad_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (!inputs || !outputs) return false;
  return mse_grad_check(inputs, num_inputs, outputs, num_outputs);
}

static size_t mse_grad_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

static aicf::Status mse_grad_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;

  if (!mse_grad_check(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& T = inputs[1];
  TensorDesc& G = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return aicf::Status::InvalidArgument;

  float scale = 0.0f;
  const bool has_scale = attr_get_f32(attr, "scale", &scale);

  if (!has_scale) {
    scale = 2.0f / (float)numel;
  }

  const int block = 256;
  const int grid = (int)((numel + block - 1) / block);

  mse_grad_impl::mse_grad_f32_kernel<<<grid, block, 0, stream>>>(
      (const float*)P.data,
      (const float*)T.data,
      (float*)G.data,
      numel,
      scale);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_mse_grad_f32_variant() {
  KernelVariant v{};
  v.name = "mse_grad_f32";
  v.priority = 0;
  v.flags = 0;
  v.launch = mse_grad_launch;
  v.supported = mse_grad_supported;
  v.query_workspace = mse_grad_workspace;
  return v;
}

} // namespace aicf::cuda
