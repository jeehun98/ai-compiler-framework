#pragma once

#include "aicf/backends/cuda/registry/registry.hpp"

namespace aicf::cuda {

// NOTE: Status enum 이름은 네 프로젝트에 맞춰 조정 필요할 수 있음.
inline aicf::Status dispatch(
    OpKind kind,
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* workspace, size_t workspace_bytes,
    cudaStream_t stream) {
  const auto& vs = KernelRegistry::instance().variants(kind);

  for (const auto& v : vs) {
    if (v.supported(inputs, num_inputs, outputs, num_outputs, attr)) {
      return v.launch(inputs, num_inputs, outputs, num_outputs,
                      attr, workspace, workspace_bytes, stream);
    }
  }

  // 아래 값 이름을 네 Status에 맞춰 수정:
  return aicf::Status::kNotSupported;
}

}  // namespace aicf::cuda
