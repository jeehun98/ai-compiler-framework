#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API (optional; not used by op_call path)
#include <aicf/backends/cuda/ops/gemm/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>

// common shim
#include "aicf/backends/cuda/ops/_common/shim/launch.hpp"
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"
#include "aicf/backends/cuda/ops/_common/shim/validate.hpp"

#include "kernels.cuh"

#include <string_view>

namespace aicf::cuda {

using namespace nvcuda;

// -------------------------
// kernels (implementation lives here in your style)
// -------------------------
namespace gemm_impl {

// -------------------------
// f32 naive kernels (existing)
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

// B is stored as [N,K] with arbitrary stride, and we interpret it as transposed:
//   logical B^T has shape [K,N], where B^T[kk, col] = B[col, kk]
__global__ void gemm_f32_naive_transB_kernel(const float* __restrict__ A,
                                            const float* __restrict__ B, // stored as [N,K]
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
// ✅ NEW: WMMA TensorCore GEMM
// - A, B: half (row-major storage as per contract)
// - C: float row-major
// - transA/transB handled by interpreting A/B storage shapes:
//     transA=false: A=[M,K], ldA=K
//     transA=true : A=[K,M], ldA=M
//     transB=false: B=[K,N], ldB=N
//     transB=true : B=[N,K], ldB=K
// - One warp computes one 16x16 tile.
// - Uses SMEM to present:
//     smemA as row_major 16x16
//     smemB as col_major 16x16
// -------------------------
__global__ void gemm_f16_tc_wmma_kernel(const __half* __restrict__ A,
                                       const __half* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K,
                                       bool transA, bool transB) {
  // 1 warp per block
  const int lane = threadIdx.x & 31;

  // tile coords
  const int tile_m = (int)blockIdx.y;
  const int tile_n = (int)blockIdx.x;
  const int m0 = tile_m * 16;
  const int n0 = tile_n * 16;

  __shared__ __half smemA[16 * 16];
  __shared__ __half smemB[16 * 16];
  __shared__ float  smemC[16 * 16];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  // leading dims for row-major storage
  const int ldA = transA ? M : K;
  const int ldB = transB ? K : N;

  // bounds for storage shapes
  // if transA: A is [K,M], else [M,K]
  const int A_rows = transA ? K : M;
  const int A_cols = transA ? M : K;
  // if transB: B is [N,K], else [K,N]
  const int B_rows = transB ? N : K;
  const int B_cols = transB ? K : N;

  for (int k0 = 0; k0 < K; k0 += 16) {
    // ---- fill smemA in row_major (i,j) ----
    for (int t = lane; t < 256; t += 32) {
      const int i = t / 16;  // 0..15
      const int j = t % 16;  // 0..15

      // logical A element needed: A_logical[m0+i, k0+j]
      // storage:
      //  - transA=false: A_storage[m0+i, k0+j]
      //  - transA=true : A_storage[k0+j, m0+i]  (since stored as [K,M])
      int a_r = 0, a_c = 0;
      if (!transA) { a_r = m0 + i; a_c = k0 + j; }
      else         { a_r = k0 + j; a_c = m0 + i; }

      __half v = __float2half(0.0f);
      if ((unsigned)a_r < (unsigned)A_rows && (unsigned)a_c < (unsigned)A_cols) {
        v = A[a_r * ldA + a_c];
      }
      smemA[i * 16 + j] = v;
    }

    // ---- fill smemB in col_major: store (i,j) -> [j,i] ----
    // logical B element needed: B_logical[k0+i, n0+j]
    // storage:
    //  - transB=false: B_storage[k0+i, n0+j]  (stored [K,N])
    //  - transB=true : B_storage[n0+j, k0+i]  (stored [N,K])
    for (int t = lane; t < 256; t += 32) {
      const int i = t / 16; // 0..15 (k)
      const int j = t % 16; // 0..15 (n)

      int b_r = 0, b_c = 0;
      if (!transB) { b_r = k0 + i; b_c = n0 + j; }
      else         { b_r = n0 + j; b_c = k0 + i; }

      __half v = __float2half(0.0f);
      if ((unsigned)b_r < (unsigned)B_rows && (unsigned)b_c < (unsigned)B_cols) {
        v = B[b_r * ldB + b_c];
      }

      // col_major store: (i,j) goes to [j,i]
      smemB[j * 16 + i] = v;
    }

    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_frag;

    wmma::load_matrix_sync(a_frag, smemA, 16);
    wmma::load_matrix_sync(b_frag, smemB, 16);
    wmma::mma_sync(acc, a_frag, b_frag, acc);

    __syncthreads();
  }

  // write accumulator to shared, then global
  wmma::store_matrix_sync(smemC, acc, 16, wmma::mem_row_major);
  __syncthreads();

  for (int t = lane; t < 256; t += 32) {
    const int i = t / 16;
    const int j = t % 16;
    const int row = m0 + i;
    const int col = n0 + j;
    if ((unsigned)row < (unsigned)M && (unsigned)col < (unsigned)N) {
      C[row * N + col] = smemC[i * 16 + j];
    }
  }
}

} // namespace gemm_impl

// -------------------------
// public API implementation (keep as-is; not used by op_call path)
// -------------------------
aicf::Status gemm_f32(const float* A,
                      const float* B,
                      float* C,
                      int M, int N, int K,
                      aicf::Stream stream) {
  if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
    return aicf::Status::InvalidArgument;
  }

  cudaStream_t s = aicf::cuda::shim::to_cuda_stream(stream);

  dim3 block(16, 16, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y,
            1);

  gemm_impl::gemm_f32_naive_kernel<<<grid, block, 0, s>>>(A, B, C, M, N, K);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

// -------------------------
// Attr helpers (local, minimal)
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

static inline bool is_f32_2d(const TensorDesc& T) {
  return (T.dtype == DType::kF32) && (T.rank() == 2);
}
static inline bool is_f16_2d(const TensorDesc& T) {
  return (T.dtype == DType::kF16) && (T.rank() == 2);
}

// strict contiguous row-major for 2D
static inline bool is_contig_2d(const TensorDesc& T) {
  if (T.rank() != 2) return false;
  // row-major contiguous: stride[1]=1, stride[0]=shape[1]
  return (T.stride[1] == 1) && (T.stride[0] == T.shape[1]);
}

// -------------------------
// Variant #1: f32 naive (existing, keep behavior)
// -------------------------
static inline bool gemm_f32_variant_check_2d_ex(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    bool transB) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& C = outputs[0];

  // A, C: strict contig (baseline)
  if (!aicf::cuda::shim::is_f32_contig_2d(A)) return false;
  if (!aicf::cuda::shim::is_f32_contig_2d(C)) return false;

  // B: f32 2D required
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
  if (!inputs || !outputs) return false;
  const bool transB = attr_get_bool(attr, "transB", false);
  return gemm_f32_variant_check_2d_ex(inputs, num_inputs, outputs, num_outputs, transB);
}

static size_t gemm_f32_variant_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

static aicf::Status gemm_f32_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;

  const bool transB = attr_get_bool(attr, "transB", false);

  if (!gemm_f32_variant_check_2d_ex(inputs, num_inputs, outputs, num_outputs, transB)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& C = outputs[0];

  const int M = static_cast<int>(A.shape[0]);
  const int K = static_cast<int>(A.shape[1]);
  const int N = static_cast<int>(C.shape[1]);

  dim3 block(16, 16, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y,
            1);

  if (!transB) {
    gemm_impl::gemm_f32_naive_kernel<<<grid, block, 0, stream>>>(
        (const float*)A.data,
        (const float*)B.data,
        (float*)C.data,
        M, N, K);
  } else {
    const int64_t bs0 = B.stride[0];
    const int64_t bs1 = B.stride[1];
    gemm_impl::gemm_f32_naive_transB_kernel<<<grid, block, 0, stream>>>(
        (const float*)A.data,
        (const float*)B.data,
        (float*)C.data,
        M, N, K, bs0, bs1);
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

// -------------------------
// ✅ Variant #2: f16 TensorCore WMMA (new)
// Contract (baseline):
//   - C: [M,N] f32 contiguous
//   - A:
//       transA=false: [M,K] f16 contiguous
//       transA=true : [K,M] f16 contiguous  (storage-transposed)
//   - B:
//       transB=false: [K,N] f16 contiguous
//       transB=true : [N,K] f16 contiguous  (storage-transposed)
// Attr semantics:
//   - transA : bool (default false)
//   - transB : bool (default false)
// Notes:
//   - This avoids stride-view transpose in v0.
//   - Works for NN/TN/NT needed for backward.
// -------------------------
static inline bool gemm_tc_check_2d_ex(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    bool transA, bool transB) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& C = outputs[0];

  if (!is_f16_2d(A) || !is_f16_2d(B) || !is_f32_2d(C)) return false;
  if (!is_contig_2d(A) || !is_contig_2d(B) || !is_contig_2d(C)) return false;

  const int64_t M = C.shape[0];
  const int64_t N = C.shape[1];
  if (M <= 0 || N <= 0) return false;

  int64_t K = -1;

  if (!transA) {
    // A: [M,K]
    if (A.shape[0] != M) return false;
    K = A.shape[1];
  } else {
    // A: [K,M]
    if (A.shape[1] != M) return false;
    K = A.shape[0];
  }

  if (K <= 0) return false;

  if (!transB) {
    // B: [K,N]
    if (B.shape[0] != K || B.shape[1] != N) return false;
  } else {
    // B: [N,K]
    if (B.shape[0] != N || B.shape[1] != K) return false;
  }

  return true;
}

static bool gemm_tc_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  if (!inputs || !outputs) return false;
  const bool transA = attr_get_bool(attr, "transA", false);
  const bool transB = attr_get_bool(attr, "transB", false);
  return gemm_tc_check_2d_ex(inputs, num_inputs, outputs, num_outputs, transA, transB);
}

static size_t gemm_tc_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

static aicf::Status gemm_tc_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;

  const bool transA = attr_get_bool(attr, "transA", false);
  const bool transB = attr_get_bool(attr, "transB", false);

  if (!gemm_tc_check_2d_ex(inputs, num_inputs, outputs, num_outputs, transA, transB)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& C = outputs[0];

  const int M = (int)C.shape[0];
  const int N = (int)C.shape[1];
  const int K = (int)(!transA ? A.shape[1] : A.shape[0]);

  dim3 block(32, 1, 1); // 1 warp
  dim3 grid((N + 15) / 16, (M + 15) / 16, 1);

  gemm_impl::gemm_f16_tc_wmma_kernel<<<grid, block, 0, stream>>>(
      (const __half*)A.data,
      (const __half*)B.data,
      (float*)C.data,
      M, N, K, transA, transB);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_gemm_f16_tc_wmma_variant() {
  KernelVariant v{};
  v.name = "gemm_f16_tc_wmma";
  v.priority = 10; // tc > naive
  v.flags = 0;
  v.launch = gemm_tc_launch;
  v.supported = gemm_tc_supported;
  v.query_workspace = gemm_tc_workspace;
  return v;
}

} // namespace aicf::cuda
