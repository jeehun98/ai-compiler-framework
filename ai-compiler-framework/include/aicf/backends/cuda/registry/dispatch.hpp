#pragma once

#include <cuda_runtime.h>
#include "aicf/core/status.hpp"
#include "aicf/backends/cuda/registry/registry.hpp"

namespace aicf::cuda {

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
  // 네 Status에는 NotSupported가 없으니 NotImplemented로 반환
  return aicf::Status::NotImplemented;
}

} // namespace aicf::cuda
