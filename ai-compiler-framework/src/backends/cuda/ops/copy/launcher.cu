#include <aicf/backends/cuda/ops/copy/api.hpp>

#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

#include <cuda_fp16.h>
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
// copy_v0 (contract-compatible)
// ============================================================
aicf::Status copy_v0(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (num_inputs != 1 || num_outputs != 1) return aicf::Status::InvalidArgument;

  const TensorDesc& X = inputs[0];
  TensorDesc& Y = outputs[0];

  if (X.dtype != Y.dtype) return aicf::Status::InvalidArgument;
  if (!same_shape(X, Y))  return aicf::Status::InvalidArgument;
  if (!is_contig(X) || !is_contig(Y)) return aicf::Status::NotImplemented;

  const int64_t n = numel(X);
  if (n <= 0) return aicf::Status::Ok;

  const int threads = 256;
  const int blocks  = (int)((n + threads - 1) / threads);

  if (X.dtype == DType::kF32) {
    const float* x = (const float*)X.data;
    float* y = (float*)Y.data;
    copy_1d_kernel<float><<<blocks, threads, 0, stream>>>(x, y, n);
    return aicf::Status::Ok;
  }

  if (X.dtype == DType::kF16) {
    const __half* x = (const __half*)X.data;
    __half* y = (__half*)Y.data;
    copy_1d_kernel<__half><<<blocks, threads, 0, stream>>>(x, y, n);
    return aicf::Status::Ok;
  }

  return aicf::Status::NotImplemented;
}

// ============================================================
// KernelVariant supported/query/launch
// ============================================================
static bool supported_copy_f32(
    const TensorDesc* in, int ni,
    const TensorDesc* out, int no,
    const void* /*attr*/) {

  if (ni != 1 || no != 1) return false;
  if (in[0].dtype != DType::kF32) return false;
  if (out[0].dtype != DType::kF32) return false;
  if (!same_shape(in[0], out[0])) return false;
  if (!is_contig(in[0]) || !is_contig(out[0])) return false;
  return true;
}

static bool supported_copy_f16(
    const TensorDesc* in, int ni,
    const TensorDesc* out, int no,
    const void* /*attr*/) {

  if (ni != 1 || no != 1) return false;
  if (in[0].dtype != DType::kF16) return false;
  if (out[0].dtype != DType::kF16) return false;
  if (!same_shape(in[0], out[0])) return false;
  if (!is_contig(in[0]) || !is_contig(out[0])) return false;
  return true;
}

static bool supported_copy_f16_vec2(
    const TensorDesc* in, int ni,
    const TensorDesc* out, int no,
    const void* attr) {

  if (!supported_copy_f16(in, ni, out, no, attr)) return false;
  const int64_t n = numel(in[0]);
  return ((n & 1) == 0);  // even numel
}

static size_t query_ws_copy(
    const TensorDesc* /*inputs*/, int /*num_inputs*/,
    const void* /*attr*/) {
  return 0; // no workspace
}

static aicf::Status launch_copy(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {

  return copy_v0(inputs, num_inputs, outputs, num_outputs, attr, workspace, workspace_bytes, stream);
}

// ============================================================
// Factories (linkage used by register_all.cpp)
// ============================================================
KernelVariant make_copy_f32_variant() {
  KernelVariant kv{};
  kv.name = "copy_f32_v0";
  kv.priority = 0;
  kv.query_workspace = query_ws_copy;
  kv.supported = supported_copy_f32;
  kv.launch = launch_copy;
  return kv;
}

KernelVariant make_copy_f16_variant() {
  KernelVariant kv{};
  kv.name = "copy_f16_v0";
  kv.priority = 0;
  kv.query_workspace = query_ws_copy;
  kv.supported = supported_copy_f16;
  kv.launch = launch_copy;
  return kv;
}

KernelVariant make_copy_f16_vec2_variant() {
  KernelVariant kv{};
  kv.name = "copy_f16_vec2_v0";
  kv.priority = 1; // prefer vec2 if supported
  kv.query_workspace = query_ws_copy;
  kv.supported = supported_copy_f16_vec2;
  kv.launch = launch_copy;
  return kv;
}

} // namespace aicf::cuda
