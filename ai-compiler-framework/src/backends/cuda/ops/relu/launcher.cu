#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

#include <aicf/backends/cuda/ops/relu/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

// -------------------------
// kernels
// -------------------------
namespace relu_impl {

__global__ void relu_f32_kernel(const float* __restrict__ in,
                               float* __restrict__ out,
                               int N) {
  const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < N) {
    const float x = in[i];
    out[i] = (x > 0.0f) ? x : 0.0f;
  }
}

} // namespace relu_impl

// -------------------------
// helpers
// -------------------------
static inline cudaStream_t to_cuda_stream(aicf::Stream s) {
  return (s.handle == nullptr) ? (cudaStream_t)0 : (cudaStream_t)s.handle;
}

// -------------------------
// public API implementation
// -------------------------
aicf::Status relu_f32(const float* in,
                      float* out,
                      int N,
                      aicf::Stream stream) {
  if (!in || !out || N <= 0) return aicf::Status::InvalidArgument;

  cudaStream_t s = to_cuda_stream(stream);

  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  relu_impl::relu_f32_kernel<<<blocks, threads, 0, s>>>(in, out, N);

  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? aicf::Status::Ok : aicf::Status::Error;
}

// -------------------------
// Registry Variant (no attr)
// Contract:
//   inputs[0]=in [N], outputs[0]=out [N]
//   contiguous + dtype F32 + rank1 only
// -------------------------
static aicf::Status relu_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (num_inputs != 1 || num_outputs != 1) return aicf::Status::InvalidArgument;

  const TensorDesc& I = inputs[0];
  TensorDesc& O = outputs[0];

  if (I.dtype != DType::F32 || O.dtype != DType::F32)
    return aicf::Status::InvalidArgument;

  if (!I.contiguous || !O.contiguous)
    return aicf::Status::InvalidArgument;

  if (I.ndim != 1 || O.ndim != 1)
    return aicf::Status::InvalidArgument;

  const int N = (int)O.shape[0];
  if ((int)I.shape[0] != N)
    return aicf::Status::InvalidArgument;

  aicf::Stream s{};
  s.handle = (void*)stream;

  return aicf::cuda::relu_f32(
      (const float*)I.data,
      (float*)O.data,
      N,
      s);
}

static bool relu_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& I = inputs[0];
  const TensorDesc& O = outputs[0];

  if (I.dtype != DType::F32 || O.dtype != DType::F32) return false;
  if (!I.contiguous || !O.contiguous) return false;
  if (I.ndim != 1 || O.ndim != 1) return false;

  return I.shape[0] == O.shape[0];
}

static size_t relu_variant_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

KernelVariant make_relu_f32_variant() {
  KernelVariant v;
  v.name = "relu_f32_naive";
  v.launch = relu_variant_launch;
  v.supported = relu_variant_supported;
  v.query_workspace = relu_variant_workspace;
  return v;
}

} // namespace aicf::cuda
