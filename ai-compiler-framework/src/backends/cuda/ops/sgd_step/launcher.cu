#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/sgd_step/api.hpp>

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
// public API
// -------------------------
aicf::Status sgd_step_f32(float* param,
                          const float* grad,
                          int64_t numel,
                          float lr,
                          aicf::Stream stream) {
  if (!param || !grad || numel <= 0) return aicf::Status::InvalidArgument;

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  const int block = 256;
  const int grid = (int)((numel + block - 1) / block);

  sgd_step_impl::sgd_step_f32_kernel<<<grid, block, 0, s>>>(param, grad, numel, lr);
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
// inputs[0] = param f32 contig, rank>=1
// inputs[1] = grad  f32 contig, same shape
// outputs[0]= param f32 contig, same shape
//
// Attr:
//   lr (optional, f32), default 1e-3
//
// Notes:
// - in-place update is allowed: outputs[0].data == inputs[0].data
// -------------------------
static inline bool sgd_step_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& P = inputs[0];
  const TensorDesc& G = inputs[1];
  const TensorDesc& O = outputs[0];

  if (!is_f32_contig(P) || !is_f32_contig(G) || !is_f32_contig(O)) return false;
  if (P.rank() < 1) return false;
  if (!same_shape(P, G)) return false;
  if (!same_shape(P, O)) return false;

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return false;

  return true;
}

static bool sgd_step_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (!inputs || !outputs) return false;
  return sgd_step_check(inputs, num_inputs, outputs, num_outputs);
}

static size_t sgd_step_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

static aicf::Status sgd_step_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;
  if (!sgd_step_check(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& P = inputs[0];
  const TensorDesc& G = inputs[1];
  TensorDesc& O = outputs[0];

  int64_t numel = 0;
  if (!compute_numel(P, &numel)) return aicf::Status::InvalidArgument;

  float lr = 1e-3f;
  if (attr) {
    const auto* pack = static_cast<const aicf::cuda::AttrPack*>(attr);
    (void)pack->get_f32("lr", &lr);
  }

  const int block = 256;
  const int grid = (int)((numel + block - 1) / block);

  // allow in-place: O.data may equal P.data
  sgd_step_impl::sgd_step_f32_kernel<<<grid, block, 0, stream>>>(
      (float*)O.data,
      (const float*)G.data,
      numel,
      lr);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_sgd_step_f32_variant() {
  KernelVariant v{};
  v.name = "sgd_step_f32";
  v.priority = 0;
  v.flags = 0;
  v.launch = sgd_step_launch;
  v.supported = sgd_step_supported;
  v.query_workspace = sgd_step_workspace;
  return v;
}

} // namespace aicf::cuda
