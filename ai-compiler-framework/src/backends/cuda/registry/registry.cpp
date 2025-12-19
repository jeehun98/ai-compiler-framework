#include "aicf/backends/cuda/registry/registry.hpp"

namespace aicf::cuda {

KernelRegistry& KernelRegistry::instance() {
  static KernelRegistry inst;
  return inst;
}

void KernelRegistry::register_kernel(OpKind kind, KernelVariant v) {
  const int k = static_cast<int>(kind);
  if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return;
  table_[k].push_back(v);
}

const std::vector<KernelVariant>& KernelRegistry::variants(OpKind kind) const {
  const int k = static_cast<int>(kind);
  static const std::vector<KernelVariant> empty{};
  if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return empty;
  return table_[k];
}

aicf::Status Dispatch(const OpCall& call) {
  // ---- basic validate (v0.1) ----
  if (call.num_inputs < 0 || call.num_outputs < 0) return aicf::Status::InvalidArgument;
  if (call.num_inputs > 0 && !call.inputs) return aicf::Status::InvalidArgument;
  if (call.num_outputs > 0 && !call.outputs) return aicf::Status::InvalidArgument;

  const int k = static_cast<int>(call.kind);
  if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return aicf::Status::InvalidArgument;

  // ---- attrs normalize ----
  const AttrPack empty{};
  const AttrPack* attrs = call.attrs ? call.attrs : &empty;
  const void* attr_ptr = static_cast<const void*>(attrs); // KernelVariant ABI: void*

  // ---- variants lookup ----
  const auto& vars = KernelRegistry::instance().variants(call.kind);
  if (vars.empty()) return aicf::Status::InvalidArgument;

  // ---- pick first supported ----
  const KernelVariant* chosen = nullptr;
  for (const auto& v : vars) {
    if (!v.supported || !v.launch) continue;
    if (v.supported(call.inputs, call.num_inputs,
                    call.outputs, call.num_outputs,
                    attr_ptr)) {
      chosen = &v;
      break;
    }
  }
  if (!chosen) return aicf::Status::NotImplemented;

  // ---- workspace policy (v0.1: must be 0) ----
  if (chosen->query_workspace) {
    const size_t ws = chosen->query_workspace(call.inputs, call.num_inputs, attr_ptr);
    if (ws != 0) return aicf::Status::NotImplemented;
  }

  // ---- launch (workspace disabled) ----
  return chosen->launch(
      call.inputs, call.num_inputs,
      call.outputs, call.num_outputs,
      attr_ptr,
      /*workspace=*/nullptr, /*workspace_bytes=*/0,
      call.stream);
}

} // namespace aicf::cuda
