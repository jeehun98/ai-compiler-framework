#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/bias_add/api.hpp>

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
aicf::Status bias_add_f32(const float* Y,
                          const float* bias,
                          float* Out,
                          int M, int N,
                          aicf::Stream stream) {
  if (!Y || !bias || !Out || M <= 0 || N <= 0) {
    return aicf::Status::InvalidArgument;
  }

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  dim3 block(16, 16, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y,
            1);

  bias_add_impl::bias_add_f32_kernel<<<grid, block, 0, s>>>(Y, bias, Out, M, N);
  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// Attr helpers (local, minimal)
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

static inline bool bias_add_check(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    int64_t axis) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& O = outputs[0];

  // v0: keep strict + simple
  // Y,O: f32 contig 2D, B: f32 contig 1D
  if (!aicf::cuda::shim::is_f32_contig_2d(Y)) return false;
  if (!aicf::cuda::shim::is_f32_contig_2d(O)) return false;

  if (!(B.dtype == DType::kF32 && B.rank() == 1 && B.contiguous)) return false;

  const int64_t M = Y.shape[0];
  const int64_t N = Y.shape[1];
  if (M <= 0 || N <= 0) return false;

  // output must match Y
  if (O.shape[0] != M || O.shape[1] != N) return false;

  // axis policy: for now only axis=1 supported (bias is length N)
  if (axis != 1) return false;

  // bias shape
  if (B.shape[0] != N) return false;

  return true;
}

// -------------------------
// Registry Variant
//
// Contract:
//   inputs[0]=Y [M,N] contig f32
//   inputs[1]=bias [N] contig f32
//   outputs[0]=Out [M,N] contig f32
//
// Attr semantics (minimal):
//   axis : int (default 1)   // only 1 supported in v0
// -------------------------

static bool bias_add_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  if (!inputs || !outputs) return false;

  int64_t axis = 1;
  (void)attr_get_i64(attr, "axis", &axis);
  return bias_add_check(inputs, num_inputs, outputs, num_outputs, axis);
}

static size_t bias_add_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

static aicf::Status bias_add_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;

  int64_t axis = 1;
  (void)attr_get_i64(attr, "axis", &axis);

  if (!bias_add_check(inputs, num_inputs, outputs, num_outputs, axis)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& Y = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& O = outputs[0];

  const int M = static_cast<int>(Y.shape[0]);
  const int N = static_cast<int>(Y.shape[1]);

  dim3 block(16, 16, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y,
            1);

  bias_add_impl::bias_add_f32_kernel<<<grid, block, 0, stream>>>(
      (const float*)Y.data,
      (const float*)B.data,
      (float*)O.data,
      M, N);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_bias_add_f32_variant() {
  KernelVariant v{};
  v.name = "bias_add_f32";
  v.priority = 0;
  v.flags = 0;
  v.launch = bias_add_launch;
  v.supported = bias_add_supported;
  v.query_workspace = bias_add_workspace;
  return v;
}

} // namespace aicf::cuda
