#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>

#include <aicf/backends/cuda/ops/adam_step/api.hpp>
#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

static inline Status cuda_to_status(cudaError_t e) {
    return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}
static inline Status cuda_last_status() {
    return cuda_to_status(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// ✅ [수정] 스키마 ID 엔디언 교정
// Python struct.pack("<ffff", ...)으로 'ADAM' 식별자를 보낼 때 매칭되는 값
// ----------------------------------------------------------------------------
static constexpr uint32_t kSchema_ADAM = 0x4144414Du; // 'ADAM' (Python 송신값에 맞춤)

struct AdamAttrs {
    float lr;
    float beta1;
    float beta2;
    float eps;
};

static inline AdamAttrs get_adam_attrs_from_attr(const AttrBlob& a) {
    AdamAttrs out{1e-3f, 0.9f, 0.999f, 1e-8f};
    if (a.schema_id == 0) return out;
    if (a.schema_id != kSchema_ADAM) return out;
    if (!a.data || a.bytes < 16) return out;

    const float* p = static_cast<const float*>(a.data);
    out.lr    = p[0];
    out.beta1 = p[1];
    out.beta2 = p[2];
    out.eps   = p[3];
    return out;
}

// ----------------------------------------------------------------------------
// Tensor Helpers
// ----------------------------------------------------------------------------
static inline int64_t numel(const TensorDesc& d) {
    int64_t n = 1;
    for (int i = 0; i < d.rank(); ++i) n *= (int64_t)d.shape[i];
    return n;
}

static inline bool same_shape(const TensorDesc& a, const TensorDesc& b) {
    if (a.rank() != b.rank()) return false;
    for (int i = 0; i < a.rank(); ++i) if (a.shape[i] != b.shape[i]) return false;
    return true;
}

// ✅ [수정] Rank 0 스칼라와 Rank 1 (size 1) 텐서 모두 허용하도록 유연하게 변경
static inline bool is_scalar_f32_flexible(const TensorDesc& d) {
    if (d.dtype != DType::kF32 || !d.contiguous) return false;
    if (d.rank() == 0) return true;
    if (d.rank() == 1 && d.shape[0] == 1) return true;
    return false;
}

static inline bool ptr_eq(const TensorDesc& a, const TensorDesc& b) { return a.data == b.data; }

// ----------------------------------------------------------------------------
// Adam Step Logic
// ----------------------------------------------------------------------------
static bool adam_step_check(const TensorDesc* in, int ni, const TensorDesc* out, int no) {
    if (ni != 6 || no != 3) return false;

    // Inputs: P, G, M, V, BC1, BC2 / Outputs: Pout, Mout, Vout
    if (in[0].dtype != DType::kF32 || in[1].dtype != DType::kF32 ||
        in[2].dtype != DType::kF32 || in[3].dtype != DType::kF32) return false;

    if (!is_scalar_f32_flexible(in[4]) || !is_scalar_f32_flexible(in[5])) return false;

    // Shape 매칭 확인
    if (!same_shape(in[0], in[1]) || !same_shape(in[0], in[2]) || !same_shape(in[0], in[3])) return false;
    if (!same_shape(in[0], out[0]) || !same_shape(in[2], out[1]) || !same_shape(in[3], out[2])) return false;

    return true;
}

static bool adam_step_f32_supported(const TensorDesc* in, int ni, const TensorDesc* out, int no, const void*) {
    return adam_step_check(in, ni, out, no);
}

static Status adam_step_f32_launch(const TensorDesc* in, int ni, TensorDesc* out, int no, const void* attr, void*, size_t, cudaStream_t stream) {
    if (!adam_step_check(in, ni, out, no) || !attr) return Status::InvalidArgument;

    const int64_t n = numel(in[0]);
    if (n <= 0) return Status::Ok;

    const AttrBlob& ab = *static_cast<const AttrBlob*>(attr);
    const AdamAttrs a = get_adam_attrs_from_attr(ab);

    // Out-of-place 대응: Pout이 P와 다르면 먼저 복사
    if (!ptr_eq(out[0], in[0])) {
        cudaMemcpyAsync(out[0].data, in[0].data, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    constexpr int kThreads = 256;
    int blocks = (int)((n + kThreads - 1) / kThreads);

    adam_step_f32_kernel_v2<<<blocks, kThreads, 0, stream>>>(
        (float*)out[0].data, (const float*)in[1].data, (const float*)in[2].data, (const float*)in[3].data,
        (float*)out[1].data, (float*)out[2].data, n,
        a.lr, a.beta1, a.beta2, a.eps,
        (const float*)in[4].data, (const float*)in[5].data);

    return cuda_last_status();
}

KernelVariant make_adam_step_f32_variant() {
    KernelVariant v{};
    v.name = "adam_step_f32_v2";
    v.kernel_id = "adam_step_f32_v2_v0"; // ✅ [수정] ID 명시
    v.priority = 0;
    v.expected_attr_schema_id = kSchema_ADAM; // ✅ [수정] 0x4144414Du 매칭
    v.launch = adam_step_f32_launch;
    v.supported = adam_step_f32_supported;
    return v;
}

// ----------------------------------------------------------------------------
// Kernel Implementation
// ----------------------------------------------------------------------------
__global__ void adam_step_f32_kernel_v2(
    float* __restrict__ Pout, const float* __restrict__ G, const float* __restrict__ M, const float* __restrict__ V,
    float* __restrict__ Mout, float* __restrict__ Vout, int64_t n,
    float lr, float beta1, float beta2, float eps,
    const float* __restrict__ bc1, const float* __restrict__ bc2) {

    const float bc1v = bc1 ? bc1[0] : 1.0f;
    const float bc2v = bc2 ? bc2[0] : 1.0f;

    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = G[i];
        float m_new = beta1 * M[i] + (1.0f - beta1) * g;
        float v_new = beta2 * V[i] + (1.0f - beta2) * (g * g);

        Mout[i] = m_new;
        Vout[i] = v_new;

        float m_hat = m_new / bc1v;
        float v_hat = v_new / bc2v;
        Pout[i] -= lr * (m_hat / (sqrtf(v_hat) + eps));
    }
}

} // namespace aicf::cuda