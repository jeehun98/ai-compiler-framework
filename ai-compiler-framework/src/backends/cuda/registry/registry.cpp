#include "aicf/backends/cuda/registry/registry.hpp"
#include "aicf/backends/cuda/registry/dispatch.hpp"

#include <mutex>
#include <vector>
#include <string>

#include "aicf/backends/cuda/registry/attr_blob.hpp"
#include "aicf/backends/cuda/registry/register_all.hpp"

namespace aicf::cuda {

// ------------------------------
// kernel_id lookup helper (NEW)
// ------------------------------
static Status DispatchById(const OpCall& call, const char* kernel_id) {
  KernelRegistry::ensure_registered();

  if (!kernel_id || kernel_id[0] == '\0') return Status::InvalidArgument;

  // basic validate
  if (call.num_inputs < 0 || call.num_outputs < 0) return Status::InvalidArgument;
  if (call.num_inputs > 0 && !call.inputs) return Status::InvalidArgument;
  if (call.num_outputs > 0 && !call.outputs) return Status::InvalidArgument;

  const int k = static_cast<int>(call.kind);
  if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return Status::InvalidArgument;

  // attrs normalize
  static const AttrBlob kEmptyAttrBlob{};
  const AttrBlob* attrs = call.attrs ? static_cast<const AttrBlob*>(call.attrs) : &kEmptyAttrBlob;
  const void* attr_ptr = static_cast<const void*>(attrs);

  // ✅ registry에서 kernel_id로 직접 찾기
  const KernelVariant* chosen = KernelRegistry::instance().find_by_id(call.kind, kernel_id);
  if (!chosen) return Status::NotImplemented;
  if (!chosen->supported || !chosen->launch) return Status::NotImplemented;

  // schema gate (0 means "no gate")
  if (chosen->expected_attr_schema_id != 0 && attrs->schema_id != chosen->expected_attr_schema_id) {
    return Status::InvalidArgument;
  }

  // supported check는 "결정 박제" 관점에서는 보통 생략 가능하지만,
  // 안전을 위해 유지(불일치면 InvalidArgument)
  if (!chosen->supported(call.inputs, call.num_inputs,
                         call.outputs, call.num_outputs,
                         attr_ptr)) {
    return Status::InvalidArgument;
  }

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

KernelRegistry& KernelRegistry::instance() {
  static KernelRegistry inst;
  return inst;
}

void KernelRegistry::ensure_registered() {
  static std::once_flag once;
  std::call_once(once, []() { aicf_cuda_register_all_kernels(); });
}

void KernelRegistry::register_kernel(OpKind kind, KernelVariant v) {
  const int k = static_cast<int>(kind);
  if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return;

  // ✅ kernel_id 필수 강제 (결정 박제의 핵심)
  if (!v.kernel_id || v.kernel_id[0] == '\0') {
    // 조용히 무시하면 디버깅 지옥. 개발 중엔 assert/throw 권장.
    return;
  }

  std::lock_guard<std::mutex> lock(mu_);

  auto entry = std::make_unique<KernelVariant>(v);

  auto& vec = table_[k];
  auto it = vec.begin();
  for (; it != vec.end(); ++it) {
    const KernelVariant* cur = it->get();
    if (!cur) continue;
    if (entry->priority > cur->priority) break;
  }

  // 실제 삽입
  KernelVariant* inserted_ptr = entry.get();
  vec.insert(it, std::move(entry));

  // ✅ by_id_ 인덱싱
  auto& mp = by_id_[k];
  std::string key(inserted_ptr->kernel_id);

  auto found = mp.find(key);
  if (found == mp.end()) {
    mp.emplace(std::move(key), inserted_ptr);
  } else {
    // 충돌 처리 정책:
    // - 보수적: 기존 유지 (stable)
    // - 공격적: 새로 덮어쓰기 (build 순서에 따라 바뀜)
    // 여기서는 기존 유지.
  }
}

const KernelVariant* KernelRegistry::find_by_id(OpKind kind, const char* kernel_id) const {
  const int k = static_cast<int>(kind);
  if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return nullptr;
  if (!kernel_id || kernel_id[0] == '\0') return nullptr;

  std::lock_guard<std::mutex> lock(mu_);
  const auto& mp = by_id_[k];

  auto it = mp.find(std::string(kernel_id));
  if (it == mp.end()) return nullptr;
  return it->second;
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

  // attrs normalize
  static const AttrBlob kEmptyAttrBlob{};
  const AttrBlob* attrs = call.attrs ? static_cast<const AttrBlob*>(call.attrs) : &kEmptyAttrBlob;
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

// Minimal C ABI-style wrapper used by bindings (legacy)
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

// ✅ NEW: Minimal C ABI-style wrapper used by bindings (kernel_id path)
Status dispatch_by_id_v0(
    OpKind kind,
    const char* kernel_id,
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

  return DispatchById(call, kernel_id);
}

} // namespace aicf::cuda
