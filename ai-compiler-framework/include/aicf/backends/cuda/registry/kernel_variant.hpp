#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "aicf/backends/cuda/registry/status.hpp"
#include "aicf/backends/cuda/registry/tensor_desc.hpp"

namespace aicf::cuda {

// flags (optional)
enum KernelVariantFlags : uint32_t {
  kKV_None        = 0,
  kKV_CaptureSafe = 1u << 0,  // reserved
};

// v0.x KernelVariant contract:
// - supported(): cheap feasibility check (shape/dtype/rank/attrs)
// - query_workspace(): optional; if present may return >0
// - launch(): invoked with workspace ptr/bytes; v0.x policy may still pass nullptr/0
//
// Selection policy:
// - higher priority wins
// - if equal priority, registry insertion order is tie-breaker (stable)
struct KernelVariant {
  const char* name = nullptr;

  // higher = earlier selection
  int priority = 0;

  // reserved for future policies (arch/capture_safe/etc.)
  uint32_t flags = 0;

  // Optional: attr schema filtering (0 = accept any).
  uint32_t expected_attr_schema_id = 0;

  Status (*launch)(
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
