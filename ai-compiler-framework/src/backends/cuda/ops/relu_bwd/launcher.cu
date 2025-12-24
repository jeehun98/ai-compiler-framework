#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/relu_bwd/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

// common shim
#include "aicf/backends/cuda/ops/_common/shim/launch.hpp"
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"
#include "aicf/backends/cuda/ops/_common/shim/validate.hpp"

#include "kernels.cuh"

namespace aicf::cuda {

// -------------------------
// public API implementation
// -------------------------
aicf::Status relu_bwd_f32(const float* Y,
                          const float* dOut,
                          float* dY,
                          int64_t numel,
                          aicf::Stream stream) {
  if (!Y || !dOut || !dY || numel <= 0) {
    return aicf::Status::InvalidArgument;
  }

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  const int block = 256;
  const int grid = (int)((numel + block - 1) / block);

  relu_bwd_impl::relu_bwd_f32_kernel<<<grid, block, 0, s>>>(Y, dOut, dY, numel);
  return aicf::cuda::shim::cuda_last_error_to_status();
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
// inputs[0] = Y    f32 contig, rank>=1  (ReLU forward output)
// inputs[1] = dOut f32 contig, same shape
// outputs[0]= dY   f32 contig, same shape
// -------------------------
static inline bool relu_bwd_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& Y = inputs[0];
  const TensorDesc& dOut = inputs[1];
  const TensorDesc& dY = outputs[0];

  if (!is_f32_contig(Y) || !is_f32_contig(dOut) || !is_f32_contig(dY)) return false;
  if (Y.rank() < 1) return false;
  if (!same_shape(Y, dOut)) return false;
  if (!same_shape(Y, dY)) return false;

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return false;

  return true;
}

static bool relu_bwd_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (!inputs || !outputs) return false;
  return relu_bwd_check(inputs, num_inputs, outputs, num_outputs);
}

static size_t relu_bwd_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

static aicf::Status relu_bwd_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;

  if (!relu_bwd_check(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& dOut = inputs[1];
  TensorDesc& dY = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(Y, &numel)) return aicf::Status::InvalidArgument;

  const int block = 256;
  const int grid = (int)((numel + block - 1) / block);

  relu_bwd_impl::relu_bwd_f32_kernel<<<grid, block, 0, stream>>>(
      (const float*)Y.data,
      (const float*)dOut.data,
      (float*)dY.data,
      numel);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_relu_bwd_f32_variant() {
  KernelVariant v{};
  v.name = "relu_bwd_f32";
  v.priority = 0;
  v.flags = 0;
  v.launch = relu_bwd_launch;
  v.supported = relu_bwd_supported;
  v.query_workspace = relu_bwd_workspace;
  return v;
}

} // namespace aicf::cuda
