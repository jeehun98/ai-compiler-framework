// ============================================================================
// src/backends/cuda/ops/gemm/launcher.cu
// - KEEP "kernel definitions inside launcher.cu" structure
// - Unify transA/transB by stride-swapped logical views (MatView2D)
// - f32: single strided naive kernel (covers NN/TN/NT/TT via view)
// - f16 TC: WMMA kernel uses GLOBAL->SMEM packing (transpose-at-load) helpers
//   * acc float, store half
// - NOTE: TC path currently requires C contiguous row-major (Ccs==1, Crs==N)
//         A/B may be strided logically (correct), perf may vary.
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

// ============================================================================
// Attr helpers
// ============================================================================
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

// ============================================================================
// Tensor helpers
// ============================================================================
static inline bool is_2d(const TensorDesc& T) { return T.rank() == 2; }
static inline bool is_f32_2d(const TensorDesc& T) { return (T.dtype == DType::kF32) && is_2d(T); }
static inline bool is_f16_2d(const TensorDesc& T) { return (T.dtype == DType::kF16) && is_2d(T); }

// stride is in ELEMENTS (torch-style), must be positive for our kernels.
static inline bool stride_valid_2d(const TensorDesc& T) {
  if (!is_2d(T)) return false;
  return (T.stride[0] > 0 && T.stride[1] > 0);
}

static inline bool is_contig_rowmajor_2d(const TensorDesc& T) {
  if (!is_2d(T)) return false;
  if (!stride_valid_2d(T)) return false;
  return (T.stride[1] == 1) && (T.stride[0] == T.shape[1]);
}

// Logical 2D view (in elements)
struct MatView2D {
  const void* data{nullptr};
  int64_t rows{0};   // logical rows
  int64_t cols{0};   // logical cols
  int64_t rs{0};     // row-stride
  int64_t cs{0};     // col-stride
  // NOTE: requires DType::kUnknown exist in your DType enum.
  // If not, either add it or set a safe default (e.g., kF32) and override below.
  DType dtype{DType::kUnknown};
};

static inline MatView2D make_view_2d(const TensorDesc& T, bool trans) {
  MatView2D v{};
  v.data  = T.data;
  v.dtype = T.dtype;

  // physical (r,c) -> r*stride0 + c*stride1
  if (!trans) {
    v.rows = T.shape[0];
    v.cols = T.shape[1];
    v.rs   = T.stride[0];
    v.cs   = T.stride[1];
  } else {
    // logical transpose: (r,c) maps to physical (c,r)
    v.rows = T.shape[1];
    v.cols = T.shape[0];
    v.rs   = T.stride[1];
    v.cs   = T.stride[0];
  }
  return v;
}

// Unified GEMM shape/stride check using logical views
static inline bool gemm_check_2d(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    bool transA, bool transB,
    DType dtypeA, DType dtypeB, DType dtypeC) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A0 = inputs[0];
  const TensorDesc& B0 = inputs[1];
  const TensorDesc& C0 = outputs[0];

  if (A0.rank()!=2 || B0.rank()!=2 || C0.rank()!=2) return false;
  if (A0.dtype!=dtypeA || B0.dtype!=dtypeB || C0.dtype!=dtypeC) return false;

  if (!stride_valid_2d(A0) || !stride_valid_2d(B0) || !stride_valid_2d(C0)) return false;

  MatView2D A = make_view_2d(A0, transA);  // [M,K]
  MatView2D B = make_view_2d(B0, transB);  // [K,N]
  MatView2D C = make_view_2d(C0, false);   // [M,N]

  const int64_t M = A.rows;
  const int64_t K = A.cols;
  const int64_t N = B.cols;

  if (M<=0 || N<=0 || K<=0) return false;
  if (B.rows != K) return false;
  if (C.rows != M || C.cols != N) return false;

  if (A.rs<=0 || A.cs<=0 || B.rs<=0 || B.cs<=0 || C.rs<=0 || C.cs<=0) return false;

  return true;
}

// ============================================================================
// kernels (definitions)  -- keep here
// ============================================================================
namespace gemm_impl {

// -------------------------
// f32 naive: unified strided
// -------------------------
__global__ void gemm_f32_naive_strided_kernel(
    const float* __restrict__ A, int64_t Ars, int64_t Acs,
    const float* __restrict__ B, int64_t Brs, int64_t Bcs,
    float* __restrict__ C, int64_t Crs, int64_t Ccs,
    int M, int N, int K) {

  const int row = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  const int col = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= M || col >= N) return;

  float acc = 0.0f;
  for (int kk = 0; kk < K; ++kk) {
    const float a = A[(int64_t)row * Ars + (int64_t)kk * Acs];
    const float b = B[(int64_t)kk * Brs + (int64_t)col * Bcs];
    acc += a * b;
  }
  C[(int64_t)row * Crs + (int64_t)col * Ccs] = acc;
}

// -------------------------
// WMMA helpers
// -------------------------
__device__ __forceinline__ int ceil16_i(int x) { return (x + 15) & ~15; }

__device__ __forceinline__
__half load_h(const __half* base, int64_t rs, int64_t cs,
             int64_t rows, int64_t cols, int r, int c) {
  if ((unsigned)r < (unsigned)rows && (unsigned)c < (unsigned)cols) {
    return base[(int64_t)r * rs + (int64_t)c * cs];
  }
  return __float2half(0.0f);
}

// -------------------------
// GLOBAL->SMEM packing (transpose-at-load)
// -------------------------
__device__ __forceinline__
void pack_smemA_rowmajor_16x16(
    __half* smemA,
    const __half* A, int64_t Ars, int64_t Acs, int64_t Am, int64_t Ak,
    int m0, int k0, int lane)
{
  // smemA[i,j] = A[m0+i, k0+j] (row-major)
  for (int t = lane; t < 256; t += 32) {
    const int i = t / 16;
    const int j = t % 16;
    smemA[i * 16 + j] = load_h(A, Ars, Acs, Am, Ak, m0 + i, k0 + j);
  }
}

__device__ __forceinline__
void pack_smemB_colmajor_16x16(
    __half* smemB,
    const __half* B, int64_t Brs, int64_t Bcs, int64_t Bk, int64_t Bn,
    int k0, int n0, int lane)
{
  // We want smemB stored in COL-MAJOR so that wmma::load_matrix_sync(..., col_major) works.
  // col-major means element (row=i,col=j) stored at smemB[j*ld + i].
  // Target: Btile[i,j] = B[k0+i, n0+j]
  // Store:  smemB[j,i] = Btile[i,j]
  for (int t = lane; t < 256; t += 32) {
    const int i = t / 16;
    const int j = t % 16;
    smemB[j * 16 + i] = load_h(B, Brs, Bcs, Bk, Bn, k0 + i, n0 + j);
  }
}

// -------------------------
// WMMA core (strided A/B, C stride-aware store)
// - computes: C[M,N] = A[M,K] @ B[K,N]
// - packs A row-major, packs B col-major (transpose-at-load)
// - acc float, store half
// -------------------------
__device__ __forceinline__
void wmma_core_out_f16_strided_packed(
    const __half* A, int64_t Ars, int64_t Acs, int64_t Am, int64_t Ak,
    const __half* B, int64_t Brs, int64_t Bcs, int64_t Bk, int64_t Bn,
    __half* C, int64_t Crs, int64_t Ccs, int64_t Cm, int64_t Cn) {

  const int lane = threadIdx.x & 31;
  const int m0 = (int)blockIdx.y * 16;
  const int n0 = (int)blockIdx.x * 16;

  __shared__ __half smemA[256]; // 16x16 row-major
  __shared__ __half smemB[256]; // 16x16 col-major
  __shared__ float  smemC[256]; // 16x16 row-major output

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  const int K = (int)Ak;
  const int K16 = ceil16_i(K);

  for (int k0 = 0; k0 < K16; k0 += 16) {
    // GLOBAL -> SMEM packing
    pack_smemA_rowmajor_16x16(smemA, A, Ars, Acs, Am, Ak, m0, k0, lane);
    pack_smemB_colmajor_16x16(smemB, B, Brs, Bcs, Bk, Bn, k0, n0, lane);
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

  // store out (stride-aware, though TC variant may restrict C to contiguous row-major)
  for (int t = lane; t < 256; t += 32) {
    const int i = t / 16;
    const int j = t % 16;
    const int r = m0 + i;
    const int c = n0 + j;
    if ((unsigned)r < (unsigned)Cm && (unsigned)c < (unsigned)Cn) {
      C[(int64_t)r * Crs + (int64_t)c * Ccs] = __float2half(smemC[i * 16 + j]);
    }
  }
}

// -------------------------
// WMMA kernel (strided)
// -------------------------
__global__ void gemm_f16_tc_wmma_out_f16_strided_kernel(
    const __half* __restrict__ A, int64_t Ars, int64_t Acs, int64_t Am, int64_t Ak,
    const __half* __restrict__ B, int64_t Brs, int64_t Bcs, int64_t Bk, int64_t Bn,
    __half* __restrict__ C, int64_t Crs, int64_t Ccs, int64_t Cm, int64_t Cn) {

  wmma_core_out_f16_strided_packed(
      A, Ars, Acs, Am, Ak,
      B, Brs, Bcs, Bk, Bn,
      C, Crs, Ccs, Cm, Cn);
}

} // namespace gemm_impl

// ============================================================================
// f32 variant: unified for transA/transB via view
// ============================================================================
static bool gemm_f32_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  const bool transA = attr_get_bool(attr, "transA", false);
  const bool transB = attr_get_bool(attr, "transB", false);

  return gemm_check_2d(inputs, num_inputs, outputs, num_outputs,
                       transA, transB, DType::kF32, DType::kF32, DType::kF32);
}

static size_t gemm_f32_variant_workspace(const TensorDesc*, int, const void*) { return 0; }

static aicf::Status gemm_f32_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  const bool transA = attr_get_bool(attr, "transA", false);
  const bool transB = attr_get_bool(attr, "transB", false);

  if (!gemm_check_2d(inputs, num_inputs, outputs, num_outputs,
                     transA, transB, DType::kF32, DType::kF32, DType::kF32)) {
    return aicf::Status::InvalidArgument;
  }

  MatView2D A = make_view_2d(inputs[0], transA);  // [M,K]
  MatView2D B = make_view_2d(inputs[1], transB);  // [K,N]
  MatView2D C = make_view_2d(outputs[0], false);  // [M,N]

  const int M = (int)A.rows;
  const int K = (int)A.cols;
  const int N = (int)B.cols;

  dim3 block(16, 16, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y,
            1);

  gemm_impl::gemm_f32_naive_strided_kernel<<<grid, block, 0, stream>>>(
      (const float*)A.data, A.rs, A.cs,
      (const float*)B.data, B.rs, B.cs,
      (float*)C.data, C.rs, C.cs,
      M, N, K);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_gemm_f32_naive_variant() {
  KernelVariant v{};
  v.name = "gemm_f32_naive_strided";
  v.priority = 0;
  v.flags = 0;
  v.launch = gemm_f32_variant_launch;
  v.supported = gemm_f32_variant_supported;
  v.query_workspace = gemm_f32_variant_workspace;
  return v;
}

// ============================================================================
// TC out_f16 variant: unified for transA/transB via view
// ============================================================================
static bool gemm_tc_out_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  const bool transA = attr_get_bool(attr, "transA", false);
  const bool transB = attr_get_bool(attr, "transB", false);

  if (!gemm_check_2d(inputs, num_inputs, outputs, num_outputs,
                     transA, transB, DType::kF16, DType::kF16, DType::kF16)) {
    return false;
  }

  // Practical constraint: require C contiguous row-major for now.
  const TensorDesc& C0 = outputs[0];
  if (!is_contig_rowmajor_2d(C0)) return false;

  return true;
}

static size_t gemm_tc_out_f16_workspace(const TensorDesc*, int, const void*) { return 0; }

static aicf::Status gemm_tc_out_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  const bool transA = attr_get_bool(attr, "transA", false);
  const bool transB = attr_get_bool(attr, "transB", false);

  if (!gemm_check_2d(inputs, num_inputs, outputs, num_outputs,
                     transA, transB, DType::kF16, DType::kF16, DType::kF16)) {
    return aicf::Status::InvalidArgument;
  }
  if (!is_contig_rowmajor_2d(outputs[0])) {
    return aicf::Status::InvalidArgument;
  }

  MatView2D A = make_view_2d(inputs[0], transA);   // [M,K]
  MatView2D B = make_view_2d(inputs[1], transB);   // [K,N]
  MatView2D C = make_view_2d(outputs[0], false);   // [M,N]

  const int M = (int)A.rows;
  const int K = (int)A.cols;
  const int N = (int)B.cols;

  dim3 block(32, 1, 1);
  dim3 grid((N + 15) / 16, (M + 15) / 16, 1);

  gemm_impl::gemm_f16_tc_wmma_out_f16_strided_kernel<<<grid, block, 0, stream>>>(
      (const __half*)A.data, A.rs, A.cs, A.rows, A.cols,
      (const __half*)B.data, B.rs, B.cs, B.rows, B.cols,
      (__half*)C.data, C.rs, C.cs, C.rows, C.cols);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

KernelVariant make_gemm_f16_tc_wmma_out_f16_variant() {
  KernelVariant v{};
  v.name = "gemm_f16_tc_wmma_out_f16_strided";
  v.priority = 20;
  v.flags = 0;
  v.launch = gemm_tc_out_f16_launch;
  v.supported = gemm_tc_out_f16_supported;
  v.query_workspace = gemm_tc_out_f16_workspace;
  return v;
}

} // namespace aicf::cuda
