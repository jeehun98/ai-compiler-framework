#include "aicf/backends/cuda/registry/registry.hpp"

namespace aicf::cuda {

KernelRegistry& KernelRegistry::instance() {
  static KernelRegistry inst;
  return inst;
}

void KernelRegistry::register_kernel(OpKind kind, KernelVariant v) {
  table_[static_cast<int>(kind)].push_back(v);
}

const std::vector<KernelVariant>& KernelRegistry::variants(OpKind kind) const {
  return table_[static_cast<int>(kind)];
}

}  // namespace aicf::cuda
