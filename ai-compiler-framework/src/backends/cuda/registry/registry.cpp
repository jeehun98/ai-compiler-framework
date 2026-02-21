#include "aicf/backends/cuda/registry/registry.hpp"
#include "aicf/backends/cuda/registry/dispatch.hpp"

#include <mutex>
#include <vector>
#include <string>
#include <cstdio> // printf 추가

#include "aicf/backends/cuda/registry/attr_blob.hpp"
#include "aicf/backends/cuda/registry/register_all.hpp"

namespace aicf::cuda {

// ------------------------------
// kernel_id lookup helper (NEW)
// ------------------------------
static Status DispatchById(const OpCall& call, const char* kernel_id) {
    KernelRegistry::ensure_registered();

    if (!kernel_id || kernel_id[0] == '\0') return Status::InvalidArgument;
    if (call.num_inputs < 0 || call.num_outputs < 0) return Status::InvalidArgument;

    const int k = static_cast<int>(call.kind);
    if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return Status::InvalidArgument;

    static const AttrBlob kEmptyAttrBlob{};
    const AttrBlob* attrs = call.attrs ? static_cast<const AttrBlob*>(call.attrs) : &kEmptyAttrBlob;
    const void* attr_ptr = static_cast<const void*>(attrs);

    const KernelVariant* chosen = KernelRegistry::instance().find_by_id(call.kind, kernel_id);
    
    // 디버그 로그: ID로 직접 호출할 때
    if (call.kind == OpKind::CrossEntropyFwd) {
        std::printf("[AICF-DEBUG] DispatchById: kind=XENT, kernel_id=%s, found=%s\n", 
                    kernel_id, chosen ? "YES" : "NO");
    }

    if (!chosen) return Status::NotImplemented;
    if (!chosen->supported || !chosen->launch) return Status::NotImplemented;

    if (chosen->expected_attr_schema_id != 0 && attrs->schema_id != chosen->expected_attr_schema_id) {
        if (call.kind == OpKind::CrossEntropyFwd) {
            std::printf("[AICF-DEBUG] XENT Schema mismatch: Expected 0x%08x, Got 0x%08x\n", 
                        chosen->expected_attr_schema_id, attrs->schema_id);
        }
        return Status::InvalidArgument;
    }

    if (!chosen->supported(call.inputs, call.num_inputs, call.outputs, call.num_outputs, attr_ptr)) {
        if (call.kind == OpKind::CrossEntropyFwd) {
            std::printf("[AICF-DEBUG] XENT supported() returned FALSE for ID: %s\n", kernel_id);
        }
        return Status::InvalidArgument;
    }

    return chosen->launch(call.inputs, call.num_inputs, call.outputs, call.num_outputs, 
                          attr_ptr, nullptr, 0, call.stream);
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

    if (!v.kernel_id || v.kernel_id[0] == '\0') return;

    std::lock_guard<std::mutex> lock(mu_);
    auto entry = std::make_unique<KernelVariant>(v);
    auto& vec = table_[k];
    auto it = vec.begin();
    for (; it != vec.end(); ++it) {
        if (entry->priority > (*it)->priority) break;
    }
    KernelVariant* inserted_ptr = entry.get();
    vec.insert(it, std::move(entry));

    auto& mp = by_id_[k];
    mp[std::string(inserted_ptr->kernel_id)] = inserted_ptr;
}

const KernelVariant* KernelRegistry::find_by_id(OpKind kind, const char* kernel_id) const {
    const int k = static_cast<int>(kind);
    if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return nullptr;
    std::lock_guard<std::mutex> lock(mu_);
    const auto& mp = by_id_[k];
    auto it = mp.find(std::string(kernel_id));
    return (it == mp.end()) ? nullptr : it->second;
}

void KernelRegistry::variants_snapshot(OpKind kind, std::vector<const KernelVariant*>& out) const {
    out.clear();
    const int k = static_cast<int>(kind);
    if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return;
    std::lock_guard<std::mutex> lock(mu_);
    const auto& vec = table_[k];
    for (const auto& p : vec) if (p) out.push_back(p.get());
}

Status Dispatch(const OpCall& call) {
    KernelRegistry::ensure_registered();

    const int k = static_cast<int>(call.kind);
    if (k < 0 || k >= static_cast<int>(OpKind::_Count)) return Status::InvalidArgument;

    static const AttrBlob kEmptyAttrBlob{};
    const AttrBlob* attrs = call.attrs ? static_cast<const AttrBlob*>(call.attrs) : &kEmptyAttrBlob;
    const void* attr_ptr = static_cast<const void*>(attrs);

    std::vector<const KernelVariant*> vars;
    KernelRegistry::instance().variants_snapshot(call.kind, vars);

    // 디버그 로그: 변체 탐색 시작
    if (call.kind == OpKind::CrossEntropyFwd || call.kind == OpKind::ReduceSum || call.kind == OpKind::Gemm) {
        std::printf("[AICF-DEBUG] Dispatch: kind=XENT, found_variants=%zu\n", vars.size());
    }

    if (vars.empty()) return Status::NotImplemented;

    const KernelVariant* chosen = nullptr;
    for (const auto* v : vars) {
        if (!v || !v->supported || !v->launch) continue;

        if (call.kind == OpKind::CrossEntropyFwd || call.kind == OpKind::Gemm) {
            std::printf("[AICF-DEBUG] Checking variant: %s\n", v->kernel_id ? v->kernel_id : "unnamed");
        }

        if (v->expected_attr_schema_id != 0 && attrs->schema_id != v->expected_attr_schema_id) {
            if (call.kind == OpKind::CrossEntropyFwd || call.kind == OpKind::Gemm) {
                std::printf("  -> Schema mismatch (0x%08x vs 0x%08x)\n", v->expected_attr_schema_id, attrs->schema_id);
            }
            continue;
        }

        bool ok = v->supported(call.inputs, call.num_inputs, call.outputs, call.num_outputs, attr_ptr);
        if (call.kind == OpKind::CrossEntropyFwd || call.kind == OpKind::Gemm) {
            std::printf("  -> supported() call: %s\n", ok ? "TRUE" : "FALSE");
        }

        if (ok) {
            chosen = v;
            break;
        }
    }

    if (!chosen) {
        if (call.kind == OpKind::CrossEntropyFwd) {
            std::printf("[AICF-DEBUG] XENT Dispatch failed: No variant passed supported() check.\n");
        }
        return Status::NotImplemented;
    }

    return chosen->launch(call.inputs, call.num_inputs, call.outputs, call.num_outputs, 
                          attr_ptr, nullptr, 0, call.stream);
}

Status dispatch_v0(OpKind kind, const TensorDesc* inputs, int32_t num_inputs,
                   TensorDesc* outputs, int32_t num_outputs, const void* attrs, cudaStream_t stream) {
    OpCall call{kind, inputs, num_inputs, outputs, num_outputs, attrs, stream};
    return Dispatch(call);
}

Status dispatch_by_id_v0(OpKind kind, const char* kernel_id, const TensorDesc* inputs, int32_t num_inputs,
                         TensorDesc* outputs, int32_t num_outputs, const void* attrs, cudaStream_t stream) {
    OpCall call{kind, inputs, num_inputs, outputs, num_outputs, attrs, stream};
    return DispatchById(call, kernel_id);
}

} // namespace aicf::cuda