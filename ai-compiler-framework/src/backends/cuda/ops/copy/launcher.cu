#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include <aicf/backends/cuda/ops/copy/api.hpp>

#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

// -------------------------
// cuda error -> Status
// -------------------------
static inline Status cuda_to_status(cudaError_t e) {
  return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}
static inline Status cuda_last_status() {
  return cuda_to_status(cudaGetLastError());
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

static inline int choose_blocks_1d(int64_t n, int threads) {
  int64_t blocks64 = (n + threads - 1) / threads;
  if (blocks64 < 1) blocks64 = 1;
  const int64_t kMaxBlocks = 4096;
  if (blocks64 > kMaxBlocks) blocks64 = kMaxBlocks;
  return (int)blocks64;
}

static size_t copy_workspace(const TensorDesc*, int, const void*) { return 0; }

// -------------------------
// contract check (shared)
// inputs[0]=X, outputs[0]=Y
// same dtype/shape + contiguous
// -------------------------
static inline bool copy_check_dt(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    DType dt) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& X = inputs[0];
  const TensorDesc& Y = outputs[0];

  if (X.dtype != dt || Y.dtype != dt) return false;
  if (!contig_anyrank(X) || !contig_anyrank(Y)) return false;
  if (!same_shape_anyrank(X, Y)) return false;

  const int64_t n = numel_anyrank(X);
  return (n >= 0);
}

// -------------------------
// F32 variant
// -------------------------
static bool copy_f32_supported(
    const TensorDesc* in, int ni,
    const TensorDesc* out, int no,
    const void* /*attr*/) {
  return copy_check_dt(in, ni, out, no, DType::kF32);
}

static Status copy_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!copy_check_dt(inputs, num_inputs, outputs, num_outputs, DType::kF32)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& X = inputs[0];
  TensorDesc& Y = outputs[0];

  const int64_t n = numel_anyrank(X);
  if (n == 0) return Status::Ok;

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(n, kThreads);

  cudaGetLastError(); // clear
  copy_impl::copy_1d_kernel<float><<<blocks, kThreads, 0, stream>>>(
      (const float*)X.data, (float*)Y.data, n);

  return cuda_last_status();
}

KernelVariant make_copy_f32_variant() {
  KernelVariant kv{};
  kv.name = "copy_f32";
  kv.priority = 0;
  kv.flags = 0;
  kv.expected_attr_schema_id = 0; // no attr
  kv.query_workspace = copy_workspace;
  kv.supported = copy_f32_supported;
  kv.launch = copy_f32_launch;
  return kv;
}

// -------------------------
// F16 variant
// -------------------------
static bool copy_f16_supported(
    const TensorDesc* in, int ni,
    const TensorDesc* out, int no,
    const void* /*attr*/) {
  return copy_check_dt(in, ni, out, no, DType::kF16);
}

static Status copy_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!copy_check_dt(inputs, num_inputs, outputs, num_outputs, DType::kF16)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& X = inputs[0];
  TensorDesc& Y = outputs[0];

  const int64_t n = numel_anyrank(X);
  if (n == 0) return Status::Ok;

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(n, kThreads);

  cudaGetLastError(); // clear
  copy_impl::copy_1d_kernel<__half><<<blocks, kThreads, 0, stream>>>(
      (const __half*)X.data, (__half*)Y.data, n);

  return cuda_last_status();
}

KernelVariant make_copy_f16_variant() {
  KernelVariant kv{};
  kv.name = "copy_f16";
  kv.priority = 0;
  kv.flags = 0;
  kv.expected_attr_schema_id = 0; // no attr
  kv.query_workspace = copy_workspace;
  kv.supported = copy_f16_supported;
  kv.launch = copy_f16_launch;
  return kv;
}

} // namespace aicf::cuda
