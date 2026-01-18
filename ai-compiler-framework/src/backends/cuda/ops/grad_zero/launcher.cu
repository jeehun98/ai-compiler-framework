#include <cuda_runtime.h>
#include <cstdint>

#include <aicf/backends/cuda/ops/grad_zero/api.hpp>

#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>

namespace aicf::cuda {

// -------------------------
// cuda error -> Status
// -------------------------
static inline Status cuda_to_status(cudaError_t e) {
  return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}

// -------------------------
// helpers
// -------------------------
static inline int64_t numel_anyrank(const TensorDesc& T) {
  const int r = T.rank();
  if (r <= 0) return 0;
  int64_t n = 1;
  for (int i = 0; i < r; ++i) {
    const int64_t d = T.shape[i];
    if (d <= 0) return 0;
    n *= d;
  }
  return n;
}

static inline bool same_shape_anyrank(const TensorDesc& A, const TensorDesc& B) {
  if (A.rank() != B.rank()) return false;
  for (int i = 0; i < A.rank(); ++i) {
    if (A.shape[i] != B.shape[i]) return false;
  }
  return true;
}

static inline bool contig_anyrank(const TensorDesc& T) { return T.contiguous; }

static inline size_t dtype_bytes(DType dt) {
  switch (dt) {
    case DType::kF32: return 4;
    case DType::kF16: return 2;
    default: return 0;
  }
}

static size_t grad_zero_workspace(const TensorDesc*, int, const void*) { return 0; }

// -------------------------
// contract check
// inputs[0]=X, outputs[0]=Y
// - allow in-place or out-of-place
// - same dtype/shape
// - contiguous
// - f16/f32 only (MVP)
// -------------------------
static inline bool grad_zero_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& X = inputs[0];
  const TensorDesc& Y = outputs[0];

  if (X.dtype != Y.dtype) return false;
  if (!same_shape_anyrank(X, Y)) return false;
  if (!contig_anyrank(X) || !contig_anyrank(Y)) return false;

  if (dtype_bytes(X.dtype) == 0) return false;

  const int64_t n = numel_anyrank(X);
  return (n >= 0);
}

// -------------------------
// supported / launch
// -------------------------
static bool grad_zero_supported(
    const TensorDesc* in, int ni,
    const TensorDesc* out, int no,
    const void* /*attr*/) {
  return grad_zero_check(in, ni, out, no);
}

static Status grad_zero_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!grad_zero_check(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  TensorDesc& Y = outputs[0];

  const int64_t n = numel_anyrank(Y);
  if (n == 0) return Status::Ok;

  const size_t eb = dtype_bytes(Y.dtype);
  const size_t bytes = (size_t)n * eb;

  // clear previous error and memset
  cudaGetLastError();
  cudaError_t e = cudaMemsetAsync(Y.data, 0, bytes, stream);
  if (e != cudaSuccess) return cuda_to_status(e);

  // also catch async launch parameter errors
  return cuda_to_status(cudaGetLastError());
}

// -------------------------
// factory
// -------------------------
KernelVariant make_grad_zero_variant() {
  KernelVariant kv{};
  kv.name = "grad_zero";
  kv.priority = 0;
  kv.flags = 0;
  kv.expected_attr_schema_id = 0; // no attr
  kv.query_workspace = grad_zero_workspace;
  kv.supported = grad_zero_supported;
  kv.launch = grad_zero_launch;
  return kv;
}

} // namespace aicf::cuda
