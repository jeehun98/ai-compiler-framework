#include <cuda_runtime.h>

#include "aicf/backends/cuda/registry/kernel_variant.hpp"
#include "aicf/backends/cuda/registry/tensor_desc.hpp"
#include "aicf/core/status.hpp"

#include "aicf/backends/cuda/ops/gemm/api.hpp"  // 네 public gemm api

namespace aicf::cuda {

static aicf::Status gemm_f32_naive_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (num_inputs != 2 || num_outputs != 1) {
    return aicf::Status::kInvalidArgument;
  }

  const auto* a = static_cast<const aicf::cuda::GemmAttr*>(attr); // 네 attr 이름/네임스페이스에 맞춰 수정
  (void)a;

  // TODO: 기존 gemm naive 커널/런처 호출
  // launch_gemm_f32_naive(A,B,C,M,N,K,alpha,beta,stream);

  (void)stream;
  return aicf::Status::kSuccess;
}

static bool gemm_f32_naive_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  if (inputs[0].dtype != DType::F32 || inputs[1].dtype != DType::F32) return false;
  if (outputs[0].dtype != DType::F32) return false;

  if (!inputs[0].contiguous || !inputs[1].contiguous || !outputs[0].contiguous) return false;

  // transA/transB는 초기엔 false만 지원하는 정책
  const auto* a = static_cast<const aicf::cuda::GemmAttr*>(attr);
  if (a && (a->transA || a->transB)) return false;

  // shape 검증은 너의 TensorDesc 규칙에 맞춰 추가
  // 예: A [M,K], B [K,N], C [M,N]

  return true;
}

static size_t gemm_f32_naive_workspace(
    const TensorDesc* /*inputs*/, int /*num_inputs*/,
    const void* /*attr*/) {
  return 0;
}

KernelVariant make_gemm_f32_naive_variant() {
  KernelVariant v;
  v.name = "gemm_f32_naive";
  v.launch = gemm_f32_naive_launch;
  v.supported = gemm_f32_naive_supported;
  v.query_workspace = gemm_f32_naive_workspace;
  return v;
}

}  // namespace aicf::cuda
