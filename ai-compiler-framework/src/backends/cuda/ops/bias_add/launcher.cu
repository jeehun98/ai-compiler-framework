#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/bias_add/api.hpp>

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
aicf::Status bias_add_f32(const float* Y,
                          const float* bias,
                          float* Out,
                          int M, int N,
                          aicf::Stream stream) {
  if (!Y || !bias || !Out || M <= 0 || N <= 0) {
    return aicf::Status::InvalidArgument;
  }

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  dim3 block(16, 16, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y,
            1);

  bias_add_impl::bias_add_f32_kernel<<<grid, block, 0, s>>>(Y, bias, Out, M, N);
  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// Attr helpers (local, minimal)
// -------------------------
static inline bool attr_get_i64(const void* attr, const char* key, int64_t* out_val) {
  if (!attr) return false;
  const auto* pack = static_cast<const aicf::cuda::AttrPack*>(attr);
  if (!pack->items || pack->size <= 0) return false;

  const std::string_view k(key);
  for (int i = 0; i < pack->size; ++i) {
    const auto& kv = pack->items[i];
    if (!kv.key) continue;
    if (std::string_view(kv.key) == k) {
      if (kv.val.tag == aicf::cuda::AttrTag::kI64) {
        *out_val = kv.val.i64;
        return true;
      }
      return false;
    }
  }
  return false;
}

// -------------------------
// last-dim-only helpers (safe)
// -------------------------
static inline bool is_f32_contig_rank_ge2(const TensorDesc& T) {
  return (T.dtype == DType::kF32) && T.contiguous && (T.rank() >= 2);
}

static inline bool axis_is_last_dim_only(const TensorDesc& Y, int64_t axis_raw) {
  const int64_t r = Y.rank();
  if (r < 2) return false;
  const int64_t last = r - 1;

  // default: axis not provided -> treat as -1 (last)
  // if provided:
  //   allow -1 or explicit last index
  if (axis_raw == -1) return true;
  if (axis_raw == last) return true;

  // allow other negative forms that resolve to last (e.g., axis = -1 only)
  // We intentionally do NOT accept -k other than -1 to keep it explicit/simple.
  return false;
}

static inline bool compute_MN_last_dim(const TensorDesc& Y, int64_t* out_M, int64_t* out_N) {
  const int64_t r = Y.rank();
  if (r < 2) return false;

  const int64_t N = Y.shape[r - 1];
  if (N <= 0) return false;

  int64_t M = 1;
  for (int64_t i = 0; i < r - 1; ++i) {
    const int64_t d = Y.shape[i];
    if (d <= 0) return false;
    M *= d;
  }

  if (M <= 0) return false;
  *out_M = M;
  *out_N = N;
  return true;
}

static inline bool bias_add_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis_raw) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& O = outputs[0];

  // Y/O: f32 contig rank>=2
  if (!is_f32_contig_rank_ge2(Y)) return false;
  if (!is_f32_contig_rank_ge2(O)) return false;

  // bias: f32 contig 1D
  if (!(B.dtype == DType::kF32 && B.rank() == 1 && B.contiguous)) return false;

  // axis policy: last dim only
  if (!axis_is_last_dim_only(Y, axis_raw)) return false;

  // output must match Y shape exactly
  if (O.rank() != Y.rank()) return false;
  for (int64_t i = 0; i < Y.rank(); ++i) {
    if (O.shape[i] != Y.shape[i]) return false;
  }

  // compute M,N (flatten all but last into M)
  int64_t M = 0, N = 0;
  if (!compute_MN_last_dim(Y, &M, &N)) return false;

  // bias length must match last dim
  if (B.shape[0] != N) return false;

  return true;
}

// -------------------------
// Registry Variant
//
// Contract (safe generalized):
//   inputs[0]=Y  f32 contig rank>=2
//   inputs[1]=bias f32 contig [N] (1D), where N = Y.shape[last]
//   outputs[0]=Out same shape as Y, f32 contig rank>=2
//
// Attr semantics:
//   axis : int (optional)
//     - allowed: -1 or (rank-1)
//     - any other value -> unsupported
//
// Flattening semantics (always safe for contiguous):
//   M = prod(Y.shape[0 .. last-1])
//   N = Y.shape[last]
// -------------------------

static bool bias_add_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  if (!inputs || !outputs) return false;

  int64_t axis = -1; // default last dim
  (void)attr_get_i64(attr, "axis", &axis);
  return bias_add_check(inputs, num_inputs, outputs, num_outputs, axis);
}

static size_t bias_add_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

static aicf::Status bias_add_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;

  int64_t axis = -1; // default last dim
  (void)attr_get_i64(attr, "axis", &axis);

  if (!bias_add_check(inputs, num_inputs, outputs, num_outputs, axis)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_last_dim(Y, &M64, &N64)) {
    return aicf::Status::InvalidArgument;
  }

  const int M = static_cast<int>(M64);
  const int N = static_cast<int>(N64);
  if (M <= 0 || N <= 0) return aicf::Status::InvalidArgument;

  dim3 block(16, 16, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y,
            1);

  bias_add_impl::bias_add_f32_kernel<<<grid, block, 0, stream>>>(
      (const float*)Y.data,
      (const float*)B.data,
      (float*)O.data,
      M, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_bias_add_f32_variant() {
  KernelVariant v{};
  v.name = "bias_add_f32";
  v.priority = 0;
  v.flags = 0;
  v.launch = bias_add_launch;
  v.supported = bias_add_supported;
  v.query_workspace = bias_add_workspace;
  return v;
}

} // namespace aicf::cuda
