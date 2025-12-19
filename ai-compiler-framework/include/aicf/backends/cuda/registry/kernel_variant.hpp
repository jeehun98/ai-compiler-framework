#pragma once
#include <cstddef>
#include <cuda_runtime.h>

#include "aicf/core/status.hpp"
#include "aicf/backends/cuda/registry/tensor_desc.hpp"

namespace aicf::cuda {

// v0.1 KernelVariant contract:
// - supported(): "이 variant가 이 call을 처리 가능?" (cheap check)
// - query_workspace(): v0.1에서는 0만 허용(또는 nullptr)
// - launch(): workspace는 v0.1에서 항상 nullptr/0로 호출됨
struct KernelVariant {
  const char* name = nullptr;

  aicf::Status (*launch)(
      const TensorDesc* inputs, int num_inputs,
      TensorDesc* outputs, int num_outputs,
      const void* attr,
      void* workspace, size_t workspace_bytes,
      cudaStream_t stream) = nullptr;

  size_t (*query_workspace)(
      const TensorDesc* inputs, int num_inputs,
      const void* attr) = nullptr;

  bool (*supported)(
      const TensorDesc* inputs, int num_inputs,
      const TensorDesc* outputs, int num_outputs,
      const void* attr) = nullptr;
};

}  // namespace aicf::cuda
