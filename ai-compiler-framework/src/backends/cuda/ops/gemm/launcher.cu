// ============================================================================
// src/backends/cuda/ops/gemm/launcher.cu
// - KEEP current "kernel definitions inside launcher.cu" structure
// - Adds f16 output variants (NN/TN/NT): A,B half, acc float, store half
// - Keeps existing f32 naive fallback
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <string_view>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

#include <aicf/backends/cuda/ops/gemm/api.hpp>

#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>

#include "aicf/backends/cuda/ops/_common/shim/launch.hpp"
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"
#include "aicf/backends/cuda/ops/_common/shim/validate.hpp"

#include "kernels.cuh"

namespace aicf::cuda {

using namespace nvcuda;

// -------------------------
// kernels (definitions)  -- keep here
// -------------------------
namespace gemm_impl {

// -------------------------
// f32 naive (existing)
// -------------------------
__global__ void gemm_f32_naive_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K) {
  const int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  const int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  const int a_row_base = row * K;
  for (int kk = 0; kk < K; ++kk) {
    acc += A[a_row_base + kk] * B[kk * N + col];
  }
  C[row * N + col] = acc;
}

__global__ void gemm_f32_naive_transB_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int M, int N, int K,
                                            int64_t B_stride0, int64_t B_stride1) {
  const int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  const int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  const int a_row_base = row * K;
  for (int kk = 0; kk < K; ++kk) {
    const float b = B[(int64_t)col * B_stride0 + (int64_t)kk * B_stride1];
    acc += A[a_row_base + kk] * b;
  }
  C[row * N + col] = acc;
}

// -------------------------
// WMMA helpers
// -------------------------
__device__ __forceinline__ int ceil16_i(int x) { return (x + 15) & ~15; }

__device__ __forceinline__ __half loadA_nn(const __half* A, int M, int K, int r, int c) {
  if ((unsigned)r < (unsigned)M && (unsigned)c < (unsigned)K) return A[r * K + c];
  return __float2half(0.0f);
}
__device__ __forceinline__ __half loadA_tn(const __half* A, int M, int K, int r, int c) {
  // A stored as [K,M] contiguous
  if ((unsigned)r < (unsigned)M && (unsigned)c < (unsigned)K) return A[c * M + r];
  return __float2half(0.0f);
}
__device__ __forceinline__ __half loadB_nn(const __half* B, int K, int N, int r, int c) {
  if ((unsigned)r < (unsigned)K && (unsigned)c < (unsigned)N) return B[r * N + c];
  return __float2half(0.0f);
}
__device__ __forceinline__ __half loadB_nt(const __half* B, int N, int K, int r, int c) {
  // B stored as [N,K] contiguous
  if ((unsigned)r < (unsigned)K && (unsigned)c < (unsigned)N) return B[c * K + r];
  return __float2half(0.0f);
}

// -------------------------
// WMMA core that stores __half output
// - acc float
// - store float -> half
// -------------------------
__device__ __forceinline__
void wmma_core_nn_out_f16(const __half* A, const __half* B, __half* C, int M, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int m0 = (int)blockIdx.y * 16;
  const int n0 = (int)blockIdx.x * 16;

  __shared__ __half smemA[256];
  __shared__ __half smemB[256];
  __shared__ float  smemC[256];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  const int K16 = ceil16_i(K);

  for (int k0 = 0; k0 < K16; k0 += 16) {
    for (int t = lane; t < 256; t += 32) {
      const int i = t / 16;
      const int j = t % 16;
      smemA[i * 16 + j] = loadA_nn(A, M, K, m0 + i, k0 + j);
      smemB[j * 16 + i] = loadB_nn(B, K, N, k0 + i, n0 + j); // col-major load layout
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, smemA, 16);
    wmma::load_matrix_sync(b_frag, smemB, 16);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncthreads();
  }

  wmma::store_matrix_sync(smemC, acc, 16, wmma::mem_row_major);
  __syncthreads();

  for (int t = lane; t < 256; t += 32) {
    const int i = t / 16;
    const int j = t % 16;
    const int r = m0 + i;
    const int c = n0 + j;
    if ((unsigned)r < (unsigned)M && (unsigned)c < (unsigned)N) {
      C[r * N + c] = __float2half(smemC[i * 16 + j]);
    }
  }
}

__device__ __forceinline__
void wmma_core_tn_out_f16(const __half* A, const __half* B, __half* C, int M, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int m0 = (int)blockIdx.y * 16;
  const int n0 = (int)blockIdx.x * 16;

  __shared__ __half smemA[256];
  __shared__ __half smemB[256];
  __shared__ float  smemC[256];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  const int K16 = ceil16_i(K);

  for (int k0 = 0; k0 < K16; k0 += 16) {
    for (int t = lane; t < 256; t += 32) {
      const int i = t / 16;
      const int j = t % 16;
      smemA[i * 16 + j] = loadA_tn(A, M, K, m0 + i, k0 + j);
      smemB[j * 16 + i] = loadB_nn(B, K, N, k0 + i, n0 + j);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, smemA, 16);
    wmma::load_matrix_sync(b_frag, smemB, 16);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncthreads();
  }

  wmma::store_matrix_sync(smemC, acc, 16, wmma::mem_row_major);
  __syncthreads();

  for (int t = lane; t < 256; t += 32) {
    const int i = t / 16;
    const int j = t % 16;
    const int r = m0 + i;
    const int c = n0 + j;
    if ((unsigned)r < (unsigned)M && (unsigned)c < (unsigned)N) {
      C[r * N + c] = __float2half(smemC[i * 16 + j]);
    }
  }
}

__device__ __forceinline__
void wmma_core_nt_out_f16(const __half* A, const __half* B, __half* C, int M, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int m0 = (int)blockIdx.y * 16;
  const int n0 = (int)blockIdx.x * 16;

  __shared__ __half smemA[256];
  __shared__ __half smemB[256];
  __shared__ float  smemC[256];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  const int K16 = ceil16_i(K);

  for (int k0 = 0; k0 < K16; k0 += 16) {
    for (int t = lane; t < 256; t += 32) {
      const int i = t / 16;
      const int j = t % 16;
      smemA[i * 16 + j] = loadA_nn(A, M, K, m0 + i, k0 + j);
      smemB[j * 16 + i] = loadB_nt(B, N, K, k0 + i, n0 + j);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;
    wmma::load_matrix_sync(a_frag, smemA, 16);
    wmma::load_matrix_sync(b_frag, smemB, 16);
    wmma::mma_sync(acc, a_frag, b_frag, acc);
    __syncthreads();
  }

  wmma::store_matrix_sync(smemC, acc, 16, wmma::mem_row_major);
  __syncthreads();

  for (int t = lane; t < 256; t += 32) {
    const int i = t / 16;
    const int j = t % 16;
    const int r = m0 + i;
    const int c = n0 + j;
    if ((unsigned)r < (unsigned)M && (unsigned)c < (unsigned)N) {
      C[r * N + c] = __float2half(smemC[i * 16 + j]);
    }
  }
}

// -------------------------
// WMMA kernels (f16->f16)
// -------------------------
__global__ void gemm_f16_tc_wmma_nn_out_f16_kernel(const __half* __restrict__ A,
                                                  const __half* __restrict__ B,
                                                  __half* __restrict__ C,
                                                  int M, int N, int K) {
  wmma_core_nn_out_f16(A, B, C, M, N, K);
}

__global__ void gemm_f16_tc_wmma_tn_out_f16_kernel(const __half* __restrict__ A,
                                                  const __half* __restrict__ B,
                                                  __half* __restrict__ C,
                                                  int M, int N, int K) {
  wmma_core_tn_out_f16(A, B, C, M, N, K);
}

__global__ void gemm_f16_tc_wmma_nt_out_f16_kernel(const __half* __restrict__ A,
                                                  const __half* __restrict__ B,
                                                  __half* __restrict__ C,
                                                  int M, int N, int K) {
  wmma_core_nt_out_f16(A, B, C, M, N, K);
}

} // namespace gemm_impl

// -------------------------
// Attr helpers
// -------------------------
static inline bool attr_get_bool(const void* attr, const char* key, bool default_val) {
  if (!attr) return default_val;
  const auto* pack = static_cast<const aicf::cuda::AttrPack*>(attr);
  if (!pack->items || pack->size <= 0) return default_val;

  const std::string_view k(key);
  for (int i = 0; i < pack->size; ++i) {
    const auto& kv = pack->items[i];
    if (!kv.key) continue;
    if (std::string_view(kv.key) == k) {
      if (kv.val.tag == aicf::cuda::AttrTag::kBool) return kv.val.b32 != 0;
      return default_val;
    }
  }
  return default_val;
}

static inline bool is_f32_2d(const TensorDesc& T) { return (T.dtype == DType::kF32) && (T.rank() == 2); }
static inline bool is_f16_2d(const TensorDesc& T) { return (T.dtype == DType::kF16) && (T.rank() == 2); }

static inline bool is_contig_2d(const TensorDesc& T) {
  if (T.rank() != 2) return false;
  if (T.stride[0] <= 0 || T.stride[1] <= 0) return false;
  return (T.stride[1] == 1) && (T.stride[0] == T.shape[1]);
}

// -------------------------
// f32 variant (fallback)
// -------------------------
static inline bool gemm_f32_variant_check_2d_ex(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    bool transB) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& C = outputs[0];

  if (!aicf::cuda::shim::is_f32_contig_2d(A)) return false;
  if (!aicf::cuda::shim::is_f32_contig_2d(C)) return false;
  if (!is_f32_2d(B)) return false;

  const int64_t M = A.shape[0];
  const int64_t K = A.shape[1];
  const int64_t N = C.shape[1];

  if (M <= 0 || N <= 0 || K <= 0) return false;
  if (C.shape[0] != M) return false;

  if (!transB) {
    if (!aicf::cuda::shim::is_f32_contig_2d(B)) return false;
    if (B.shape[0] != K || B.shape[1] != N) return false;
  } else {
    if (B.shape[0] != N || B.shape[1] != K) return false;
    if (B.stride[0] <= 0 || B.stride[1] <= 0) return false;
  }
  return true;
}

static bool gemm_f32_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {
  const bool transB = attr_get_bool(attr, "transB", false);
  return gemm_f32_variant_check_2d_ex(inputs, num_inputs, outputs, num_outputs, transB);
}

static size_t gemm_f32_variant_workspace(const TensorDesc*, int, const void*) { return 0; }

static aicf::Status gemm_f32_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  const bool transB = attr_get_bool(attr, "transB", false);

  if (!gemm_f32_variant_check_2d_ex(inputs, num_inputs, outputs, num_outputs, transB)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& C = outputs[0];

  const int M = (int)A.shape[0];
  const int K = (int)A.shape[1];
  const int N = (int)C.shape[1];

  dim3 block(16, 16, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y,
            1);

  if (!transB) {
    gemm_impl::gemm_f32_naive_kernel<<<grid, block, 0, stream>>>(
        (const float*)A.data, (const float*)B.data, (float*)C.data, M, N, K);
  } else {
    gemm_impl::gemm_f32_naive_transB_kernel<<<grid, block, 0, stream>>>(
        (const float*)A.data, (const float*)B.data, (float*)C.data,
        M, N, K, B.stride[0], B.stride[1]);
  }
  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_gemm_f32_naive_variant() {
  KernelVariant v{};
  v.name = "gemm_f32_naive";
  v.priority = 0;
  v.flags = 0;
  v.launch = gemm_f32_variant_launch;
  v.supported = gemm_f32_variant_supported;
  v.query_workspace = gemm_f32_variant_workspace;
  return v;
}

// ============================================================================
// TC out_f16 variants (NN/TN/NT)
// ============================================================================

static inline bool gemm_tc_out_f16_check_nn(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];  // [M,K]
  const TensorDesc& B = inputs[1];  // [K,N]
  const TensorDesc& C = outputs[0]; // [M,N]

  if (!is_f16_2d(A) || !is_f16_2d(B) || !is_f16_2d(C)) return false;
  if (!is_contig_2d(A) || !is_contig_2d(B) || !is_contig_2d(C)) return false;

  const int64_t M = A.shape[0];
  const int64_t K = A.shape[1];
  const int64_t N = B.shape[1];

  if (M <= 0 || N <= 0 || K <= 0) return false;
  if (B.shape[0] != K) return false;
  if (C.shape[0] != M || C.shape[1] != N) return false;
  return true;
}

static inline bool gemm_tc_out_f16_check_tn(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];  // stored [K,M]
  const TensorDesc& B = inputs[1];  // [K,N]
  const TensorDesc& C = outputs[0]; // [M,N]

  if (!is_f16_2d(A) || !is_f16_2d(B) || !is_f16_2d(C)) return false;
  if (!is_contig_2d(A) || !is_contig_2d(B) || !is_contig_2d(C)) return false;

  const int64_t K = A.shape[0];
  const int64_t M = A.shape[1];
  const int64_t N = B.shape[1];

  if (M <= 0 || N <= 0 || K <= 0) return false;
  if (B.shape[0] != K) return false;
  if (C.shape[0] != M || C.shape[1] != N) return false;
  return true;
}

static inline bool gemm_tc_out_f16_check_nt(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];  // [M,K]
  const TensorDesc& B = inputs[1];  // stored [N,K]
  const TensorDesc& C = outputs[0]; // [M,N]

  if (!is_f16_2d(A) || !is_f16_2d(B) || !is_f16_2d(C)) return false;
  if (!is_contig_2d(A) || !is_contig_2d(B) || !is_contig_2d(C)) return false;

  const int64_t M = A.shape[0];
  const int64_t K = A.shape[1];
  const int64_t N = B.shape[0];

  if (M <= 0 || N <= 0 || K <= 0) return false;
  if (B.shape[1] != K) return false;
  if (C.shape[0] != M || C.shape[1] != N) return false;
  return true;
}

static size_t gemm_tc_out_f16_workspace(const TensorDesc*, int, const void*) { return 0; }

// --- NN out_f16 ---
static bool gemm_tc_nn_out_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {
  const bool transA = attr_get_bool(attr, "transA", false);
  const bool transB = attr_get_bool(attr, "transB", false);
  if (transA || transB) return false;
  return gemm_tc_out_f16_check_nn(inputs, num_inputs, outputs, num_outputs);
}

static aicf::Status gemm_tc_nn_out_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  const bool transA = attr_get_bool(attr, "transA", false);
  const bool transB = attr_get_bool(attr, "transB", false);
  if (transA || transB) return aicf::Status::InvalidArgument;

  if (!gemm_tc_out_f16_check_nn(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& C = outputs[0];

  const int M = (int)C.shape[0];
  const int N = (int)C.shape[1];
  const int K = (int)A.shape[1];

  dim3 block(32, 1, 1);
  dim3 grid((N + 15) / 16, (M + 15) / 16, 1);

  gemm_impl::gemm_f16_tc_wmma_nn_out_f16_kernel<<<grid, block, 0, stream>>>(
      (const __half*)A.data, (const __half*)B.data, (__half*)C.data, M, N, K);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_gemm_f16_tc_wmma_nn_out_f16_variant() {
  KernelVariant v{};
  v.name = "gemm_f16_tc_wmma_nn_out_f16";
  v.priority = 20;
  v.flags = 0;
  v.launch = gemm_tc_nn_out_f16_launch;
  v.supported = gemm_tc_nn_out_f16_supported;
  v.query_workspace = gemm_tc_out_f16_workspace;
  return v;
}

// --- TN out_f16 ---
static bool gemm_tc_tn_out_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {
  const bool transA = attr_get_bool(attr, "transA", false);
  const bool transB = attr_get_bool(attr, "transB", false);
  if (!transA || transB) return false;
  return gemm_tc_out_f16_check_tn(inputs, num_inputs, outputs, num_outputs);
}

static aicf::Status gemm_tc_tn_out_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  const bool transA = attr_get_bool(attr, "transA", false);
  const bool transB = attr_get_bool(attr, "transB", false);
  if (!transA || transB) return aicf::Status::InvalidArgument;

  if (!gemm_tc_out_f16_check_tn(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0]; // stored [K,M]
  const TensorDesc& B = inputs[1]; // [K,N]
  TensorDesc& C = outputs[0];      // [M,N]

  const int M = (int)C.shape[0];
  const int N = (int)C.shape[1];
  const int K = (int)A.shape[0];

  dim3 block(32, 1, 1);
  dim3 grid((N + 15) / 16, (M + 15) / 16, 1);

  gemm_impl::gemm_f16_tc_wmma_tn_out_f16_kernel<<<grid, block, 0, stream>>>(
      (const __half*)A.data, (const __half*)B.data, (__half*)C.data, M, N, K);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_gemm_f16_tc_wmma_tn_out_f16_variant() {
  KernelVariant v{};
  v.name = "gemm_f16_tc_wmma_tn_out_f16";
  v.priority = 20;
  v.flags = 0;
  v.launch = gemm_tc_tn_out_f16_launch;
  v.supported = gemm_tc_tn_out_f16_supported;
  v.query_workspace = gemm_tc_out_f16_workspace;
  return v;
}

// --- NT out_f16 ---
static bool gemm_tc_nt_out_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {
  const bool transA = attr_get_bool(attr, "transA", false);
  const bool transB = attr_get_bool(attr, "transB", false);
  if (transA || !transB) return false;
  return gemm_tc_out_f16_check_nt(inputs, num_inputs, outputs, num_outputs);
}

static aicf::Status gemm_tc_nt_out_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  const bool transA = attr_get_bool(attr, "transA", false);
  const bool transB = attr_get_bool(attr, "transB", false);
  if (transA || !transB) return aicf::Status::InvalidArgument;

  if (!gemm_tc_out_f16_check_nt(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0]; // [M,K]
  const TensorDesc& B = inputs[1]; // stored [N,K]
  TensorDesc& C = outputs[0];      // [M,N]

  const int M = (int)C.shape[0];
  const int N = (int)C.shape[1];
  const int K = (int)A.shape[1];

  dim3 block(32, 1, 1);
  dim3 grid((N + 15) / 16, (M + 15) / 16, 1);

  gemm_impl::gemm_f16_tc_wmma_nt_out_f16_kernel<<<grid, block, 0, stream>>>(
      (const __half*)A.data, (const __half*)B.data, (__half*)C.data, M, N, K);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_gemm_f16_tc_wmma_nt_out_f16_variant() {
  KernelVariant v{};
  v.name = "gemm_f16_tc_wmma_nt_out_f16";
  v.priority = 20;
  v.flags = 0;
  v.launch = gemm_tc_nt_out_f16_launch;
  v.supported = gemm_tc_nt_out_f16_supported;
  v.query_workspace = gemm_tc_out_f16_workspace;
  return v;
}

} // namespace aicf::cuda
