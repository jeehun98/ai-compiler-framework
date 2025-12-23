#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/reduce_sum/api.hpp>

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
aicf::Status reduce_sum_lastdim_f32(const float* dY,
                                   float* dB,
                                   int M, int N,
                                   aicf::Stream stream) {
  if (!dY || !dB || M <= 0 || N <= 0) {
    return aicf::Status::InvalidArgument;
  }

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  dim3 block(256, 1, 1);
  dim3 grid((N + block.x - 1) / block.x, 1, 1);

  reduce_sum_impl::reduce_sum_lastdim_f32_kernel<<<grid, block, 0, s>>>(dY, dB, M, N);
  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// Attr helpers (same style)
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

  if (axis_raw == -1) return true;
  if (axis_raw == last) return true;
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

// -------------------------
// ReduceSum check
//
// Contract (safe generalized):
//   inputs[0]=dY  f32 contig rank>=2
//   outputs[0]=dB f32 contig rank==1, shape [N], where N = dY.shape[last]
//
// Attr semantics:
//   axis : int (optional)
//     - allowed: -1 or (rank-1)
// -------------------------
static inline bool reduce_sum_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis_raw) {

  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& dY = inputs[0];
  const TensorDesc& dB = outputs[0];

  if (!is_f32_contig_rank_ge2(dY)) return false;
  if (!(dB.dtype == DType::kF32 && dB.contiguous && dB.rank() == 1)) return false;

  if (!axis_is_last_dim_only(dY, axis_raw)) return false;

  int64_t M = 0, N = 0;
  if (!compute_MN_last_dim(dY, &M, &N)) return false;

  if (dB.shape[0] != N) return false;

  return true;
}

// -------------------------
// Registry hooks
// -------------------------
static bool reduce_sum_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  if (!inputs || !outputs) return false;

  int64_t axis = -1;
  (void)attr_get_i64(attr, "axis", &axis);
  return reduce_sum_check(inputs, num_inputs, outputs, num_outputs, axis);
}

static size_t reduce_sum_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

static aicf::Status reduce_sum_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;

  int64_t axis = -1;
  (void)attr_get_i64(attr, "axis", &axis);

  if (!reduce_sum_check(inputs, num_inputs, outputs, num_outputs, axis)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& dY = inputs[0];
  TensorDesc& dB = outputs[0];

  int64_t M64 = 0, N64 = 0;
  if (!compute_MN_last_dim(dY, &M64, &N64)) {
    return aicf::Status::InvalidArgument;
  }

  const int M = static_cast<int>(M64);
  const int N = static_cast<int>(N64);
  if (M <= 0 || N <= 0) return aicf::Status::InvalidArgument;

  dim3 block(256, 1, 1);
  dim3 grid((N + block.x - 1) / block.x, 1, 1);

  reduce_sum_impl::reduce_sum_lastdim_f32_kernel<<<grid, block, 0, stream>>>(
      (const float*)dY.data,
      (float*)dB.data,
      M, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_reduce_sum_lastdim_f32_variant() {
  KernelVariant v{};
  v.name = "reduce_sum_lastdim_f32";
  v.priority = 0;
  v.flags = 0;
  v.launch = reduce_sum_launch;
  v.supported = reduce_sum_supported;
  v.query_workspace = reduce_sum_workspace;
  return v;
}

} // namespace aicf::cuda
