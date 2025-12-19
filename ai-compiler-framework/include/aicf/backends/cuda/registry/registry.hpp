#pragma once
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "aicf/core/status.hpp"
#include "aicf/backends/cuda/registry/op_kind.hpp"
#include "aicf/backends/cuda/registry/kernel_variant.hpp"
#include "aicf/backends/cuda/registry/tensor_desc.hpp"
#include "aicf/backends/cuda/registry/attr_pack.hpp"

namespace aicf::cuda {

struct OpCall {
  OpKind kind{};
  const TensorDesc* inputs = nullptr;
  int32_t num_inputs = 0;

  TensorDesc* outputs = nullptr;
  int32_t num_outputs = 0;

  // v0.1: nullptr or AttrPack*
  const AttrPack* attrs = nullptr;

  cudaStream_t stream = nullptr;
};

class KernelRegistry {
 public:
  static KernelRegistry& instance();

  void register_kernel(OpKind kind, KernelVariant v);
  const std::vector<KernelVariant>& variants(OpKind kind) const;

 private:
  KernelRegistry() = default;

  std::vector<KernelVariant> table_[static_cast<int>(OpKind::_Count)];
};

// unified entrypoint (Plan A)
aicf::Status Dispatch(const OpCall& call);

}  // namespace aicf::cuda
