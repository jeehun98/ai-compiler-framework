#include <cuda_runtime.h>

#include "aicf/backends/cuda/registry/kernel_variant.hpp"
#include "aicf/backends/cuda/registry/tensor_desc.hpp"
#include "aicf/core/status.hpp"

// 네 eltwise public api가 이미 있다면 include
#include "aicf/backends/cuda/ops/eltwise/api.hpp"

// 너의 기존 커널/런처 선언(이미 있으면 중복 금지)
// 예: kernels.cuh에 있는 커널 런치용 함수/커널
// #include "kernels.cuh"

namespace aicf::cuda {

// ---- Add ----
static aicf::Status eltwise_add_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  // 기대: inputs[0], inputs[1] -> outputs[0]
  if (num_inputs != 2 || num_outputs != 1) {
    return aicf::Status::kInvalidArgument;
  }

  // TODO: 여기서 기존 네 add 런처/커널을 호출해라.
  // 예:
  // int N = ...; (TensorDesc에서 shape를 해석)
  // launch_add_f32((float*)inputs[0].data, (float*)inputs[1].data, (float*)outputs[0].data, N, stream);

  (void)stream;
  return aicf::Status::kSuccess;
}

static bool eltwise_add_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  // 초기 정책: f32 + contiguous only + same shape(간단)
  if (inputs[0].dtype != DType::F32 || inputs[1].dtype != DType::F32) return false;
  if (outputs[0].dtype != DType::F32) return false;

  if (!inputs[0].contiguous || !inputs[1].contiguous || !outputs[0].contiguous) return false;

  if (inputs[0].ndim != outputs[0].ndim) return false;
  if (inputs[1].ndim != outputs[0].ndim) return false;

  for (int i = 0; i < outputs[0].ndim; ++i) {
    if (inputs[0].shape[i] != outputs[0].shape[i]) return false;
    if (inputs[1].shape[i] != outputs[0].shape[i]) return false;
  }
  return true;
}

static size_t eltwise_add_f32_workspace(
    const TensorDesc* /*inputs*/, int /*num_inputs*/,
    const void* /*attr*/) {
  return 0;
}

// 외부에서 register_all이 호출할 factory
KernelVariant make_eltwise_add_f32_variant() {
  KernelVariant v;
  v.name = "add_f32_naive";
  v.launch = eltwise_add_f32_launch;
  v.supported = eltwise_add_f32_supported;
  v.query_workspace = eltwise_add_f32_workspace;
  return v;
}

// ---- ReLU ----
static aicf::Status eltwise_relu_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  // 기대: inputs[0] -> outputs[0]
  if (num_inputs != 1 || num_outputs != 1) {
    return aicf::Status::kInvalidArgument;
  }

  // TODO: 여기서 기존 네 relu 런처/커널을 호출해라.
  // launch_relu_f32((float*)inputs[0].data, (float*)outputs[0].data, N, stream);

  (void)stream;
  return aicf::Status::kSuccess;
}

static bool eltwise_relu_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (num_inputs != 1 || num_outputs != 1) return false;

  if (inputs[0].dtype != DType::F32 || outputs[0].dtype != DType::F32) return false;
  if (!inputs[0].contiguous || !outputs[0].contiguous) return false;

  if (inputs[0].ndim != outputs[0].ndim) return false;
  for (int i = 0; i < outputs[0].ndim; ++i) {
    if (inputs[0].shape[i] != outputs[0].shape[i]) return false;
  }
  return true;
}

static size_t eltwise_relu_f32_workspace(
    const TensorDesc* /*inputs*/, int /*num_inputs*/,
    const void* /*attr*/) {
  return 0;
}

KernelVariant make_eltwise_relu_f32_variant() {
  KernelVariant v;
  v.name = "relu_f32_naive";
  v.launch = eltwise_relu_f32_launch;
  v.supported = eltwise_relu_f32_supported;
  v.query_workspace = eltwise_relu_f32_workspace;
  return v;
}

}  // namespace aicf::cuda
