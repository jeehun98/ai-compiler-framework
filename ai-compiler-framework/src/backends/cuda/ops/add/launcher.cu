#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/add/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

// common shim
#include "aicf/backends/cuda/ops/_common/shim/launch.hpp"
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"
#include "aicf/backends/cuda/ops/_common/shim/validate.hpp"

// optional (future): attrs helpers
// #include "aicf/backends/cuda/ops/_common/shim/attrs.hpp"

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
// public API implementation
// -------------------------
aicf::Status add_f32(const float* a,
                     const float* b,
                     float* out,
                     int N,
                     aicf::Stream stream) {
  if (!a || !b || !out || N <= 0) return aicf::Status::InvalidArgument;

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;
  add_impl::add_f32_kernel<<<blocks, kThreads, 0, s>>>(a, b, out, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// Registry Variant - v0.2 Plan A (no workspace, no attr semantics yet)
//
// Contract:
//   inputs[0]=A [N], inputs[1]=B [N], outputs[0]=O [N]
//   binding guarantees: CUDA + contiguous (stride in desc is contiguous-by-contract)
//
// This variant supports:
//   - F32 only
// -------------------------

// v0.2: centralize the validation logic in one function used by both
// supported() and launch(). This avoids divergence between the two.
static inline bool add_variant_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& O = outputs[0];

  // v0.2: keep checks minimal + consistent (binding already did heavier checks)
  if (!aicf::cuda::shim::is_f32_contig_1d(A)) return false;
  if (!aicf::cuda::shim::is_f32_contig_1d(B)) return false;
  if (!aicf::cuda::shim::is_f32_contig_1d(O)) return false;

  if (!aicf::cuda::shim::same_shape_1d(A, B)) return false;
  if (!aicf::cuda::shim::same_shape_1d(A, O)) return false;

  // N must be positive to launch
  if (O.shape[0] <= 0) return false;

  return true;
}

static bool add_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (!inputs || !outputs) return false;
  return add_variant_check(inputs, num_inputs, outputs, num_outputs);
}

static size_t add_variant_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

static aicf::Status add_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;

  // v0.2: launch path relies on the same predicate as supported()
  if (!add_variant_check(inputs, num_inputs,
                         outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  const int N = static_cast<int>(O.shape[0]);

  // Direct kernel launch on provided cudaStream_t.
  // NOTE: This keeps stream semantics correct (binding should pass PyTorch current stream).
  constexpr int kThreads = 256;
  const int blocks = (N + kThreads - 1) / kThreads;
  add_impl::add_f32_kernel<<<blocks, kThreads, 0, stream>>>(
      (const float*)A.data,
      (const float*)B.data,
      (float*)O.data,
      N);

  return aicf::cuda::shim::cuda_last_error_to_status();

  // Alternative: call public API (extra stream wrapper hop)
  // const aicf::Stream s = aicf::cuda::shim::from_cuda_stream(stream);
  // return aicf::cuda::add_f32(
  //     (const float*)A.data,
  //     (const float*)B.data,
  //     (float*)O.data,
  //     N,
  //     s);
}

KernelVariant make_add_f32_variant() {
  KernelVariant v{};
  v.name = "add_f32_naive";
  v.priority = 0;               // v0.2: tune later (e.g. vectorized version > naive)
  v.flags = 0;                  // reserved
  v.launch = add_variant_launch;
  v.supported = add_variant_supported;
  v.query_workspace = add_variant_workspace;
  return v;
}

} // namespace aicf::cuda
