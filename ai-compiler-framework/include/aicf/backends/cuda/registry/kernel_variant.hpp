#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "aicf/core/status.hpp"
#include "aicf/backends/cuda/registry/tensor_desc.hpp"

namespace aicf::cuda {

// v0.2 KernelVariant contract:
// - supported(): cheap feasibility check (shape/dtype/rank/attrs)
// - query_workspace(): optional; if present may return >0 (future v0.3+)
// - launch(): invoked with workspace ptr/bytes; v0.1 policy may still pass nullptr/0
//
// Selection policy (suggested):
// - higher priority wins
// - if equal priority, registry order is tie-breaker
struct KernelVariant {
  const char* name = nullptr;

  // higher = earlier selection
  int priority = 0;

  // reserved for future policies (arch/capture_safe/etc.)
  uint32_t flags = 0;

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
