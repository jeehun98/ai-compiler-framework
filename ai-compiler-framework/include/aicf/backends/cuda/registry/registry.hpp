#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>

#include "aicf/backends/cuda/registry/status.hpp"
#include "aicf/backends/cuda/registry/op_kind.hpp"
#include "aicf/backends/cuda/registry/kernel_variant.hpp"
#include "aicf/backends/cuda/registry/tensor_desc.hpp"

namespace aicf::cuda {

struct OpCall {
  OpKind kind{};
  const TensorDesc* inputs = nullptr;
  int32_t num_inputs = 0;

  TensorDesc* outputs = nullptr;
  int32_t num_outputs = 0;

  // nullptr or AttrBlob* (ABI-neutral)
  const void* attrs = nullptr;

  cudaStream_t stream = nullptr;
};

class KernelRegistry {
 public:
  static KernelRegistry& instance();

  // Thread-safe
  void register_kernel(OpKind kind, KernelVariant v);

  // Snapshot pointers (stable) without holding registry lock during supported()/launch().
  void variants_snapshot(OpKind kind, std::vector<const KernelVariant*>& out) const;

  // ✅ kernel_id lookup (decision-applied path)
  const KernelVariant* find_by_id(OpKind kind, const char* kernel_id) const;

  // Optional: ensure global registration (call_once)
  static void ensure_registered();

 private:
  KernelRegistry() = default;

  using Entry = std::unique_ptr<KernelVariant>;

  std::vector<Entry> table_[static_cast<int>(OpKind::_Count)];

  // ✅ by-id index: per OpKind map(kernel_id -> KernelVariant*)
  // Pointers refer to objects owned by table_[]
  std::unordered_map<std::string, const KernelVariant*> by_id_[static_cast<int>(OpKind::_Count)];

  mutable std::mutex mu_;
};

// unified entrypoint (legacy runtime dispatch)
Status Dispatch(const OpCall& call);

} // namespace aicf::cuda
