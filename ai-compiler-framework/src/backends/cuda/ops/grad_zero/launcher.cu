#include <aicf/backends/cuda/ops/grad_zero/api.hpp>

#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

#include <cuda_runtime.h>

#include "kernels.cuh"

namespace aicf::cuda {

static inline int64_t numel(const TensorDesc& d) {
  int64_t n = 1;
  for (int i = 0; i < d.r.rank; ++i) n *= (int64_t)d.shape[i];
  return n;
}

static inline bool same_shape(const TensorDesc& a, const TensorDesc& b) {
  if (a.r.rank != b.r.rank) return false;
  for (int i = 0; i < a.r.rank; ++i) {
    if (a.shape[i] != b.shape[i]) return false;
  }
  return true;
}

static inline bool is_contig(const TensorDesc& d) { return d.contiguous; }

// ============================================================
// grad_zero_v0 (contract-compatible)
// - in-place zeroing: outputs[0] points to same buffer as inputs[0] OR separate buffer allowed.
// - MVP: contiguous only.
// - dtype-agnostic: memset bytes to 0.
// ============================================================
aicf::Status grad_zero_v0(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (num_inputs != 1 || num_outputs != 1) return aicf::Status::InvalidArgument;

  const TensorDesc& X = inputs[0];
  TensorDesc& Y = outputs[0];

  // allow in-place or out-of-place, but enforce shape/dtype match
  if (X.dtype != Y.dtype) return aicf::Status::InvalidArgument;
  if (!same_shape(X, Y))  return aicf::Status::InvalidArgument;
  if (!is_contig(X) || !is_contig(Y)) return aicf::Status::NotImplemented;

  const int64_t n = numel(X);
  if (n <= 0) return aicf::Status::Ok;

  // byte size (TensorDesc가 elem_bytes를 제공하면 그걸 쓰고, 없으면 dtype switch)
  size_t elem_bytes = 0;
  switch (X.dtype) {
    case DType::kF32: elem_bytes = 4; break;
    case DType::kF16: elem_bytes = 2; break;
    // 필요하면 여기 확장: kBF16, kI32, ...
    default: return aicf::Status::NotImplemented;
  }

  const size_t bytes = (size_t)n * elem_bytes;
  cudaError_t e = cudaMemsetAsync(Y.data, 0, bytes, stream);
  return (e == cudaSuccess) ? aicf::Status::Ok : aicf::Status::RuntimeError;
}

// ============================================================
// KernelVariant supported/query/launch
// ============================================================
static bool supported_grad_zero_contig(
    const TensorDesc* in, int ni,
    const TensorDesc* out, int no,
    const void* /*attr*/) {

  if (ni != 1 || no != 1) return false;
  if (in[0].dtype != out[0].dtype) return false;
  if (!same_shape(in[0], out[0])) return false;
  if (!is_contig(in[0]) || !is_contig(out[0])) return false;

  // MVP: f16/f32만
  if (!(in[0].dtype == DType::kF32 || in[0].dtype == DType::kF16)) return false;
  return true;
}

static size_t query_ws_grad_zero(
    const TensorDesc* /*inputs*/, int /*num_inputs*/,
    const void* /*attr*/) {
  return 0;
}

static aicf::Status launch_grad_zero(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {

  return grad_zero_v0(inputs, num_inputs, outputs, num_outputs, attr, workspace, workspace_bytes, stream);
}

// ============================================================
// Factory (linkage used by register_all.cpp)
// ============================================================
KernelVariant make_grad_zero_variant() {
  KernelVariant kv{};
  kv.name = "grad_zero_contig_v0";
  kv.priority = 0;
  kv.query_workspace = query_ws_grad_zero;
  kv.supported = supported_grad_zero_contig;
  kv.launch = launch_grad_zero;
  return kv;
}

} // namespace aicf::cuda
