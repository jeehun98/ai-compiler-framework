// ============================================================================
// src/backends/cuda/ops/gemm/launcher.cu
// - KEEP "kernel definitions inside launcher.cu" structure
// - Unify transA/transB by stride-swapped logical views (MatView2D)
// - f32: single strided naive kernel (covers NN/TN/NT/TT via view)
// - f16 TC: WMMA kernel uses GLOBAL->SMEM packing (transpose-at-load) helpers
//   * acc float, store half
// - NOTE: TC path currently requires C contiguous row-major (Ccs==1, Crs==N)
//         A/B may be strided logically (correct), perf may vary.
//
// PATCH v2:
// - Robustly read transA/transB even if AttrPack uses different key names.
// - Robustly read BOOL or I64 encoded bool.
// - Optional debug dump of AttrPack (compile with -DAICF_GEMM_DUMP_ATTR=1).
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
static inline bool attr_has_key_sv(const aicf::cuda::AttrPack* pack, std::string_view k, int* out_idx) {
  if (!pack || !pack->items || pack->size <= 0) return false;
  for (int i = 0; i < pack->size; ++i) {
    const auto& kv = pack->items[i];
    if (!kv.key) continue;
    if (std::string_view(kv.key) == k) {
      if (out_idx) *out_idx = i;
      return true;
    }
  }
  return false;
}

static inline bool attr_get_bool_i64_by_idx(const aicf::cuda::AttrPack* pack, int idx, bool* out) {
  if (!pack || !out) return false;
  if (idx < 0 || idx >= pack->size) return false;
  const auto& kv = pack->items[idx];
  // accept bool or i64
  if (kv.val.tag == aicf::cuda::AttrTag::kBool) {
    *out = (kv.val.b32 != 0);
    return true;
  }
  if (kv.val.tag == aicf::cuda::AttrTag::kI64) {
    *out = (kv.val.i64 != 0);
    return true;
  }
  return false;
}

#if defined(AICF_GEMM_DUMP_ATTR) && AICF_GEMM_DUMP_ATTR
static inline void attr_dump(const void* attr, const char* where) {
  const auto* pack = static_cast<const aicf::cuda::AttrPack*>(attr);
  if (!pack || !pack->items || pack->size <= 0) {
    printf("[gemm][attrdump][%s] (empty)\n", where);
    return;
  }
  printf("[gemm][attrdump][%s] size=%d\n", where, pack->size);
  for (int i = 0; i < pack->size; ++i) {
    const auto& kv = pack->items[i];
    const char* k = kv.key ? kv.key : "(null)";
    // tag print: you can extend this if enum changes
    int tag = (int)kv.val.tag;
    long long i64 = 0;
    int b32 = 0;
    float f32 = 0.f;
    if (kv.val.tag == aicf::cuda::AttrTag::kI64) i64 = (long long)kv.val.i64;
    if (kv.val.tag == aicf::cuda::AttrTag::kBool) b32 = (int)kv.val.b32;
    if (kv.val.tag == aicf::cuda::AttrTag::kF32) f32 = kv.val.f32;
    printf("  [%d] key=%s tag=%d (i64=%lld b32=%d f32=%f)\n", i, k, tag, i64, b32, f32);
  }
}
#else
static inline void attr_dump(const void*, const char*) {}
#endif

static inline bool attr_get_bool_or_i64_anykey(
    const void* attr,
    const char* const* keys, int nkeys,
    bool default_val) {

  const auto* pack = static_cast<const aicf::cuda::AttrPack*>(attr);
  if (!pack || !pack->items || pack->size <= 0) return default_val;

  for (int k = 0; k < nkeys; ++k) {
    int idx = -1;
    if (attr_has_key_sv(pack, std::string_view(keys[k]), &idx)) {
      bool v = default_val;
      if (attr_get_bool_i64_by_idx(pack, idx, &v)) return v;
      return default_val; // key exists but wrong tag
    }
  }
  return default_val;
}

static inline bool read_transA(const void* attr) {
  // cover common python-side naming variants
  const char* keys[] = {"transA", "trans_a", "transposeA", "ta"};
  return attr_get_bool_or_i64_anykey(attr, keys, (int)(sizeof(keys)/sizeof(keys[0])), false);
}
static inline bool read_transB(const void* attr) {
  const char* keys[] = {"transB", "trans_b", "transposeB", "tb"};
  return attr_get_bool_or_i64_anykey(attr, keys, (int)(sizeof(keys)/sizeof(keys[0])), false);
}

// ============================================================================
// Tensor helpers
// ============================================================================
static inline bool is_2d(const TensorDesc& T) { return T.rank() == 2; }
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
  int64_t rows{0};
  int64_t cols{0};
  int64_t rs{0};
  int64_t cs{0};
  DType dtype{DType::kUnknown};
};

static inline MatView2D make_view_2d(const TensorDesc& T, bool trans) {
  MatView2D v{};
  v.data  = T.data;
  v.dtype = T.dtype;

  if (!trans) {
    v.rows = T.shape[0];
    v.cols = T.shape[1];
    v.rs   = T.stride[0];
    v.cs   = T.stride[1];
  } else {
    v.rows = T.shape[1];
    v.cols = T.shape[0];
    v.rs   = T.stride[1];
    v.cs   = T.stride[0];
  }
  return v;
}

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

// WMMA helpers / pack / core 그대로 유지
__device__ __forceinline__ int ceil16_i(int x) { return (x + 15) & ~15; }

__device__ __forceinline__
__half load_h(const __half* base, int64_t rs, int64_t cs,
             int64_t rows, int64_t cols, int r, int c) {
  if ((unsigned)r < (unsigned)rows && (unsigned)c < (unsigned)cols) {
    return base[(int64_t)r * rs + (int64_t)c * cs];
  }
  return __float2half(0.0f);
}

__device__ __forceinline__
void pack_smemA_rowmajor_16x16(
    __half* smemA,
    const __half* A, int64_t Ars, int64_t Acs, int64_t Am, int64_t Ak,
    int m0, int k0, int lane)
{
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
  for (int t = lane; t < 256; t += 32) {
    const int i = t / 16;
    const int j = t % 16;
    smemB[j * 16 + i] = load_h(B, Brs, Bcs, Bk, Bn, k0 + i, n0 + j);
  }
}

__device__ __forceinline__
void wmma_core_out_f16_strided_packed(
    const __half* A, int64_t Ars, int64_t Acs, int64_t Am, int64_t Ak,
    const __half* B, int64_t Brs, int64_t Bcs, int64_t Bk, int64_t Bn,
    __half* C, int64_t Crs, int64_t Ccs, int64_t Cm, int64_t Cn) {

  const int lane = threadIdx.x & 31;
  const int m0 = (int)blockIdx.y * 16;
  const int n0 = (int)blockIdx.x * 16;

  __shared__ __half smemA[256];
  __shared__ __half smemB[256];
  __shared__ float  smemC[256];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.0f);

  const int K = (int)Ak;
  const int K16 = ceil16_i(K);

  for (int k0 = 0; k0 < K16; k0 += 16) {
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
// f32 variant
// ============================================================================
static bool gemm_f32_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  const bool transA = read_transA(attr);
  const bool transB = read_transB(attr);

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

  attr_dump(attr, "gemm_f32_launch");

  const bool transA = read_transA(attr);
  const bool transB = read_transB(attr);

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
// TC out_f16 variant
// ============================================================================
static bool gemm_tc_out_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  const bool transA = read_transA(attr);
  const bool transB = read_transB(attr);

  if (!gemm_check_2d(inputs, num_inputs, outputs, num_outputs,
                     transA, transB, DType::kF16, DType::kF16, DType::kF16)) {
    return false;
  }

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

  attr_dump(attr, "gemm_tc_f16_launch");

  const bool transA = read_transA(attr);
  const bool transB = read_transB(attr);

  if (!gemm_check_2d(inputs, num_inputs, outputs, num_outputs,
                     transA, transB, DType::kF16, DType::kF16, DType::kF16)) {
    return aicf::Status::InvalidArgument;
  }
  if (!is_contig_rowmajor_2d(outputs[0])) {
    return aicf::Status::InvalidArgument;
  }

  MatView2D A = make_view_2d(inputs[0], transA);
  MatView2D B = make_view_2d(inputs[1], transB);
  MatView2D C = make_view_2d(outputs[0], false);

  const int M = (int)A.rows;
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
