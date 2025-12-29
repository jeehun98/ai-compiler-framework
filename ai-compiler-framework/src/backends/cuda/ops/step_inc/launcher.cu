#include <cuda_runtime.h>
#include <cstdint>

#include <aicf/core/status.hpp>

#include <aicf/backends/cuda/ops/step_inc/api.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

#include "aicf/backends/cuda/ops/_common/shim/status.hpp"

namespace aicf::cuda {

__global__ void step_inc_i32_kernel(int32_t* step) {
  // single element
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    step[0] += 1;
  }
}

// contract-compatible op entry (not strictly needed but keeps symmetry)
aicf::Status step_inc_v0(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;
  if (num_inputs != 1 || num_outputs != 1) return aicf::Status::InvalidArgument;

  const TensorDesc& S = inputs[0];
  TensorDesc& O = outputs[0];

  // must be int32 scalar
  if (S.dtype != DType::kI32 || O.dtype != DType::kI32) return aicf::Status::NotImplemented;
  if (!S.contiguous || !O.contiguous) return aicf::Status::NotImplemented;

  // scalar rank=0 (shape_len=0) OR rank=1 with shape[0]==1(혹시 구현체가 그렇게 올 수도)
  // 여기선 rank=0만 요구해도 되는데, 호환성 위해 둘 다 허용.
  const bool scalar0 = (S.r.rank == 0 && O.r.rank == 0);
  const bool scalar1 = (S.r.rank == 1 && O.r.rank == 1 && S.shape[0] == 1 && O.shape[0] == 1);
  if (!(scalar0 || scalar1)) return aicf::Status::NotImplemented;

  // in-place allowed: O.data == S.data
  if (S.data != O.data) {
    // 그래도 out-buffer 정책이면 동일 버퍼로 주는게 정석이라서,
    // 여기서는 강제해버리자.
    return aicf::Status::InvalidArgument;
  }

  step_inc_i32_kernel<<<1, 32, 0, stream>>>((int32_t*)O.data);
  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// KernelVariant glue
// -------------------------
static size_t ws_step_inc(const TensorDesc*, int, const void*) { return 0; }

static bool supported_step_inc(
    const TensorDesc* in, int ni,
    const TensorDesc* out, int no,
    const void* /*attr*/) {

  if (!in || !out) return false;
  if (ni != 1 || no != 1) return false;
  if (in[0].dtype != DType::kI32 || out[0].dtype != DType::kI32) return false;
  if (!in[0].contiguous || !out[0].contiguous) return false;

  const bool scalar0 = (in[0].r.rank == 0 && out[0].r.rank == 0);
  const bool scalar1 = (in[0].r.rank == 1 && out[0].r.rank == 1 && in[0].shape[0] == 1 && out[0].shape[0] == 1);
  if (!(scalar0 || scalar1)) return false;

  // enforce in-place
  if (in[0].data != out[0].data) return false;

  return true;
}

static aicf::Status launch_step_inc(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {

  return step_inc_v0(inputs, num_inputs, outputs, num_outputs, attr, workspace, workspace_bytes, stream);
}

KernelVariant make_step_inc_variant() {
  KernelVariant kv{};
  kv.name = "step_inc_i32_v0";
  kv.priority = 0;
  kv.query_workspace = ws_step_inc;
  kv.supported = supported_step_inc;
  kv.launch = launch_step_inc;
  return kv;
}

} // namespace aicf::cuda
