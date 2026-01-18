#include "aicf/backends/cuda/registry/registry.hpp"
#include "aicf/backends/cuda/registry/dispatch.hpp"

#include <mutex>
#include <vector>

#include "aicf/backends/cuda/registry/attr_blob.hpp"
#include "aicf/backends/cuda/registry/register_all.hpp"

namespace aicf::cuda {

KernelRegistry& KernelRegistry::instance() {
  static KernelRegistry inst;
  return inst;
}

void KernelRegistry::ensure_registered() {
  static std::once_flag once;
  std::call_once(once, []() {
    aicf_cuda_register_all_kernels();
  });
}

void KernelRegistry::register_kernel(OpKind kind, KernelVariant v) {
  const int k = static_cast<int>(kind);
  if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return;

  std::lock_guard<std::mutex> lock(mu_);

  auto entry = std::make_unique<KernelVariant>(v);

  // Insert sorted by priority desc, stable among equals.
  auto& vec = table_[k];
  auto it = vec.begin();
  for (; it != vec.end(); ++it) {
    const KernelVariant* cur = it->get();
    if (!cur) continue;
    if (entry->priority > cur->priority) break;
  }
  vec.insert(it, std::move(entry));
}

void KernelRegistry::variants_snapshot(OpKind kind, std::vector<const KernelVariant*>& out) const {
  out.clear();

  const int k = static_cast<int>(kind);
  if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return;

  std::lock_guard<std::mutex> lock(mu_);
  const auto& vec = table_[k];
  out.reserve(vec.size());
  for (const auto& p : vec) {
    if (p) out.push_back(p.get());
  }
}

Status Dispatch(const OpCall& call) {
  KernelRegistry::ensure_registered();

  // basic validate
  if (call.num_inputs < 0 || call.num_outputs < 0) return Status::InvalidArgument;
  if (call.num_inputs > 0 && !call.inputs) return Status::InvalidArgument;
  if (call.num_outputs > 0 && !call.outputs) return Status::InvalidArgument;

  const int k = static_cast<int>(call.kind);
  if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return Status::InvalidArgument;

  // stream policy: explicit stream only (capture-safe)
  // if (!call.stream) return Status::InvalidArgument;

  // attrs normalize
  static const AttrBlob kEmptyAttrBlob{};
  const AttrBlob* attrs = call.attrs
      ? static_cast<const AttrBlob*>(call.attrs)
      : &kEmptyAttrBlob;
  const void* attr_ptr = static_cast<const void*>(attrs);

  // snapshot variants
  std::vector<const KernelVariant*> vars;
  KernelRegistry::instance().variants_snapshot(call.kind, vars);
  if (vars.empty()) return Status::NotImplemented;

  const KernelVariant* chosen = nullptr;
  for (const auto* v : vars) {
    if (!v || !v->supported || !v->launch) continue;

    // optional schema gate (0 means "no gate")
    if (v->expected_attr_schema_id != 0 && attrs->schema_id != v->expected_attr_schema_id) {
      continue;
    }

    if (v->supported(call.inputs, call.num_inputs,
                     call.outputs, call.num_outputs,
                     attr_ptr)) {
      chosen = v;
      break;
    }
  }
  if (!chosen) return Status::NotImplemented;

  // workspace policy (v0.x restriction)
  if (chosen->query_workspace) {
    const size_t ws = chosen->query_workspace(call.inputs, call.num_inputs, attr_ptr);
    if (ws != 0) return Status::NotImplemented;
  }

  return chosen->launch(
      call.inputs, call.num_inputs,
      call.outputs, call.num_outputs,
      attr_ptr,
      /*workspace=*/nullptr, /*workspace_bytes=*/0,
      call.stream);
}

// Minimal C ABI-style wrapper used by bindings
Status dispatch_v0(
    OpKind kind,
    const TensorDesc* inputs, int32_t num_inputs,
    TensorDesc* outputs, int32_t num_outputs,
    const void* attrs,
    cudaStream_t stream) {

  OpCall call{};
  call.kind = kind;
  call.inputs = inputs;
  call.num_inputs = num_inputs;
  call.outputs = outputs;
  call.num_outputs = num_outputs;
  call.attrs = attrs;
  call.stream = stream;

  return Dispatch(call);
}

} // namespace aicf::cuda
