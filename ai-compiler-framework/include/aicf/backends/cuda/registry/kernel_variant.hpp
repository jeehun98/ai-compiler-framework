#pragma once
#include <cstddef>
#include <cuda_runtime.h>

#include "aicf/core/status.hpp"  // 네 프로젝트 status
#include "aicf/backends/cuda/registry/tensor_desc.hpp"

namespace aicf::cuda {

struct KernelVariant {
  const char* name = nullptr;

  // 실행 함수
  aicf::Status (*launch)(
      const TensorDesc* inputs, int num_inputs,
      TensorDesc* outputs, int num_outputs,
      const void* attr,
      void* workspace, size_t workspace_bytes,
      cudaStream_t stream) = nullptr;

  // workspace 필요량
  size_t (*query_workspace)(
      const TensorDesc* inputs, int num_inputs,
      const void* attr) = nullptr;

  // 지원 조건
  bool (*supported)(
      const TensorDesc* inputs, int num_inputs,
      const TensorDesc* outputs, int num_outputs,
      const void* attr) = nullptr;
};

}  // namespace aicf::cuda
