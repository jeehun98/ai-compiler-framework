#pragma once
#include <vector>

#include "aicf/backends/cuda/registry/op_kind.hpp"
#include "aicf/backends/cuda/registry/kernel_variant.hpp"

namespace aicf::cuda {

class KernelRegistry {
 public:
  static KernelRegistry& instance();

  void register_kernel(OpKind kind, KernelVariant v);
  const std::vector<KernelVariant>& variants(OpKind kind) const;

 private:
  KernelRegistry() = default;

  std::vector<KernelVariant> table_[static_cast<int>(OpKind::_Count)];
};

}  // namespace aicf::cuda
