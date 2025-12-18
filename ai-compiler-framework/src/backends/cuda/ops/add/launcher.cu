#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

#include <aicf/backends/cuda/ops/add/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

// -------------------------
// kernels
// -------------------------
namespace add_impl {

__global__ void add_f32_kernel(const float* __restrict__ a,
                              const float* __restrict__ b,
                              float* __restrict__ out,
                              int N) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N) out[i] = a[i] + b[i];
}

} // namespace add_impl

// -------------------------
// helpers
// -------------------------
static inline cudaStream_t to_cuda_stream(aicf::Stream s) {
  return (s.handle == nullptr) ? (cudaStream_t)0 : (cudaStream_t)s.handle;
}

// -------------------------
// public API implementation
// -------------------------
aicf::Status add_f32(const float* a,
                     const float* b,
                     float* out,
                     int N,
                     aicf::Stream stream) {
  if (!a || !b || !out || N <= 0) return aicf::Status::InvalidArgument;

  cudaStream_t s = to_cuda_stream(stream);

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  add_impl::add_f32_kernel<<<blocks, threads, 0, s>>>(a, b, out, N);

  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? aicf::Status::Ok : aicf::Status::Error;
}

// -------------------------
// Registry Variant (no attr)
// Contract:
//   inputs[0]=a [N], inputs[1]=b [N], outputs[0]=out [N]
//   contiguous + dtype F32 + rank1 only
// -------------------------
static aicf::Status add_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (num_inputs != 2 || num_outputs != 1) return aicf::Status::InvalidArgument;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  if (A.dtype != DType::F32 || B.dtype != DType::F32 || O.dtype != DType::F32)
    return aicf::Status::InvalidArgument;

  if (!A.contiguous || !B.contiguous || !O.contiguous)
    return aicf::Status::InvalidArgument;

  if (A.ndim != 1 || B.ndim != 1 || O.ndim != 1)
    return aicf::Status::InvalidArgument;

  const int N = (int)O.shape[0];
  if ((int)A.shape[0] != N || (int)B.shape[0] != N)
    return aicf::Status::InvalidArgument;

  aicf::Stream s{};
  s.handle = (void*)stream;

  return aicf::cuda::add_f32(
      (const float*)A.data,
      (const float*)B.data,
      (float*)O.data,
      N,
      s);
}

static bool add_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& O = outputs[0];

  if (A.dtype != DType::F32 || B.dtype != DType::F32 || O.dtype != DType::F32) return false;
  if (!A.contiguous || !B.contiguous || !O.contiguous) return false;
  if (A.ndim != 1 || B.ndim != 1 || O.ndim != 1) return false;

  const int64_t N = O.shape[0];
  return (A.shape[0] == N) && (B.shape[0] == N);
}

static size_t add_variant_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

KernelVariant make_add_f32_variant() {
  KernelVariant v;
  v.name = "add_f32_naive";
  v.launch = add_variant_launch;
  v.supported = add_variant_supported;
  v.query_workspace = add_variant_workspace;
  return v;
}

} // namespace aicf::cuda
