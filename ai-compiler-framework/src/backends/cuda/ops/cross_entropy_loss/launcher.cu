#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <cstdio>

#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

static inline Status cuda_last_status() {
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}

namespace {
struct XentAttr { int ignore_index; int reduction; };
static constexpr uint32_t kSchemaXENT = 0x58454E54; // 'XENT' in Little-Endian

static inline XentAttr parse_attr(const void* attr) {
    XentAttr a{-100, 0}; // 기본값: ignore=-100, reduction=mean
    if (!attr) return a;
    const AttrBlob* b = (const AttrBlob*)attr;
    if (b->schema_id != kSchemaXENT || !b->data) return a;
    const int32_t* p = (const int32_t*)b->data;
    a.ignore_index = (int)p[0]; 
    a.reduction = (int)p[1];
    return a;
}
} // anon namespace

// ============================================================================
// Forward
// ============================================================================
static bool xent_fwd_supported(const TensorDesc* in, int ni, const TensorDesc* out, int no, const void*) {
    // 💡 [수정] 체크 조건을 유연하게 변경하여 NotImplemented(2) 에러를 방지합니다.
    if (ni < 2 || no < 1) return false;
    // DType만 확인하고 나머지는 Launch 단계에서 처리하도록 위임
    if (in[0].dtype != DType::kF32 || in[1].dtype != DType::kI32) return false;
    return true; 
}

static Status xent_fwd_launch(const TensorDesc* in, int ni, TensorDesc* out, int no, const void* attr, void*, size_t, cudaStream_t stream) {
    const int N = (int)in[0].shape[0];
    const int C = (int)in[0].shape[1];
    const XentAttr a = parse_attr(attr);

    // ✅ d_loss와 d_valid의 타입을 명확히 분리하여 선언 (LNK2019 및 타입 에러 방지)
    float *d_loss = nullptr; 
    int *d_valid = nullptr;

    cudaMallocAsync(&d_loss, sizeof(float), stream);
    cudaMallocAsync(&d_valid, sizeof(int), stream);
    cudaMemsetAsync(d_loss, 0, sizeof(float), stream);
    cudaMemsetAsync(d_valid, 0, sizeof(int), stream);

    // Forward 커널: 개별 손실과 유효 타겟 개수 누적
    xent_impl::xent_fwd_sum_f32<<<N, 256, 0, stream>>>(
        (const float*)in[0].data, (const int32_t*)in[1].data, N, C, a.ignore_index, d_loss, d_valid);

    // Finalize: Reduction(Mean/Sum) 적용하여 최종 출력 텐서에 기록
    xent_impl::xent_finalize_loss_f32<<<1, 1, 0, stream>>>(
        d_loss, d_valid, a.reduction, (float*)out[0].data);

    cudaFreeAsync(d_loss, stream); 
    cudaFreeAsync(d_valid, stream);
    return cuda_last_status();
}

KernelVariant make_cross_entropy_loss_fwd_f32_variant() {
    KernelVariant v{};
    v.name = "cross_entropy_loss_fwd_f32";
    v.expected_attr_schema_id = kSchemaXENT;
    v.launch = xent_fwd_launch;
    v.supported = xent_fwd_supported;
    return v;
}

// ============================================================================
// Backward
// ============================================================================
static bool xent_bwd_supported(const TensorDesc* in, int ni, const TensorDesc* out, int no, const void*) {
    if (ni < 3 || no < 1) return false;
    if (in[0].dtype != DType::kF32 || in[1].dtype != DType::kI32) return false;
    return true;
}

static Status xent_bwd_launch(const TensorDesc* in, int ni, TensorDesc* out, int no, const void* attr, void*, size_t, cudaStream_t stream) {
    const int N = (int)in[0].shape[0];
    const int C = (int)in[0].shape[1];
    const XentAttr a = parse_attr(attr);

    int* d_valid = nullptr;
    cudaMallocAsync(&d_valid, sizeof(int), stream);
    cudaMemsetAsync(d_valid, 0, sizeof(int), stream);

    if (a.reduction == 0) { // Mean 모드일 때만 실제 유효 개수를 세어줌
        xent_impl::xent_count_valid_i32<<<(N+255)/256, 256, 0, stream>>>(
            (const int32_t*)in[1].data, N, a.ignore_index, d_valid);
    } else {
        // Sum 모드: 분모를 1로 고정하여 커널 로직 재사용
        xent_impl::xent_set_int_scalar<<<1, 1, 0, stream>>>(d_valid, 1);
    }

    // Backward 커널: dLogits 계산
    xent_impl::xent_bwd_f32<<<N, 256, 0, stream>>>(
        (const float*)in[0].data, (const int32_t*)in[1].data, N, C, a.ignore_index, 
        (const float*)in[2].data, (const int*)d_valid, a.reduction, (float*)out[0].data);

    cudaFreeAsync(d_valid, stream);
    return cuda_last_status();
}

KernelVariant make_cross_entropy_loss_bwd_f32_variant() {
    KernelVariant v{};
    v.name = "cross_entropy_loss_bwd_f32";
    v.expected_attr_schema_id = kSchemaXENT;
    v.launch = xent_bwd_launch;
    v.supported = xent_bwd_supported;
    return v;
}

} // namespace aicf::cuda


// ============================================================================
// Kernel Implementations (Linking Fix: Matching headers exactly)
// ============================================================================

namespace aicf::cuda::xent_impl {

__global__ void xent_set_int_scalar(int* p, int v) {
    if (threadIdx.x == 0) p[0] = v;
}

__global__ void xent_finalize_loss_f32(
    const float* __restrict__ loss_sum,
    const int* __restrict__ valid,
    int reduction,
    float* __restrict__ out_loss) 
{
    if (threadIdx.x == 0) {
        float v = loss_sum[0];
        int n = valid[0];
        if (reduction == 0) out_loss[0] = (n > 0) ? (v / (float)n) : 0.0f;
        else                out_loss[0] = v;
    }
}

__global__ void xent_count_valid_i32(
    const int32_t* __restrict__ t,
    int N,
    int ignore_index,
    int* __restrict__ out_valid) 
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < N) {
        if (t[i] != ignore_index) atomicAdd(out_valid, 1);
    }
}

// --- Reduction Helpers ---
__inline__ __device__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}
__inline__ __device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__inline__ __device__ float block_max(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31; int wid = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    float m = (wid == 0) ? ((lane < ((blockDim.x+31)>>5)) ? sh[lane] : -CUDART_INF_F) : -CUDART_INF_F;
    if (wid == 0) m = warp_max(m);
    if (lane == 0) sh[0] = m;
    __syncthreads();
    return sh[0];
}
__inline__ __device__ float block_sum(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31; int wid = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    float s = (wid == 0) ? ((lane < ((blockDim.x+31)>>5)) ? sh[lane] : 0.0f) : 0.0f;
    if (wid == 0) s = warp_sum(s);
    if (lane == 0) sh[0] = s;
    __syncthreads();
    return sh[0];
}

__global__ void xent_fwd_sum_f32(
    const float* __restrict__ logits,
    const int32_t* __restrict__ targets,
    int N, int C,
    int ignore_index,
    float* __restrict__ out_loss_sum,
    int* __restrict__ out_valid) 
{
    int n = (int)blockIdx.x;
    if (n >= N) return;
    int t = targets[n];
    if (t == ignore_index || (unsigned)t >= (unsigned)C) return;

    const float* row = logits + (size_t)n * C;
    float m = -CUDART_INF_F;
    for (int c = threadIdx.x; c < C; c += blockDim.x) m = fmaxf(m, row[c]);
    m = block_max(m);

    float s = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) s += __expf(row[c] - m);
    s = block_sum(s);

    if (threadIdx.x == 0 && s > 0.0f) {
        float logZ = logf(s) + m;
        atomicAdd(out_loss_sum, logZ - row[t]);
        atomicAdd(out_valid, 1);
    }
}

__global__ void xent_bwd_f32(
    const float* __restrict__ logits,
    const int32_t* __restrict__ targets,
    int N, int C,
    int ignore_index,
    const float* __restrict__ grad_loss,
    const int* __restrict__ valid,
    int reduction,
    float* __restrict__ dlogits) 
{
    int n = (int)blockIdx.x;
    if (n >= N) return;
    int t = targets[n];
    float* dx = dlogits + (size_t)n * C;

    if (t == ignore_index || (unsigned)t >= (unsigned)C) {
        for (int c = threadIdx.x; c < C; c += blockDim.x) dx[c] = 0.0f;
        return;
    }

    const float* row = logits + (size_t)n * C;
    float m = -CUDART_INF_F;
    for (int c = threadIdx.x; c < C; c += blockDim.x) m = fmaxf(m, row[c]);
    m = block_max(m);

    float s = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) s += __expf(row[c] - m);
    s = block_sum(s);

    float scale = grad_loss[0];
    if (reduction == 0) {
        int vc = valid[0];
        scale /= (vc > 0 ? (float)vc : 1.0f);
    }

    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float p = __expf(row[c] - m) / s;
        dx[c] = (p - (c == t ? 1.0f : 0.0f)) * scale;
    }
}

} // namespace aicf::cuda::xent_impl