// ============================================================================
// src/backends/cuda/ops/gemm/launcher.cu  (core-free / AttrBlob)
// - kernel definitions live here (keep structure)
// - attrs are AttrBlob(schema_id + raw bytes)
// - supports f32 naive strided + f16 WMMA out_f16 (C must be contiguous row-major)
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>
#include <cstring>   // memcpy
#include <algorithm>

#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

using namespace nvcuda;

// ---- cuda error -> Status (core-free) ----
static inline Status cuda_to_status(cudaError_t e) {
  return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}
static inline Status cuda_last_status() {
  return cuda_to_status(cudaGetLastError());
}

// ============================================================================
// AttrBlob schema (GEMM)
// - schema_id == 0 => default transA=0, transB=0
// - schema_id == kAttrSchema_Gemm => parse GemmAttrV0 from bytes
// Python pack: struct.pack("<ii", trans_a, trans_b)
// ============================================================================
static constexpr uint32_t kAttrSchema_Gemm = 0x47454D4Du; // 'GEMM'

struct GemmAttrV0 {
  int32_t transA;
  int32_t transB;
};

static inline void read_gemm_attr(const void* attr, bool* out_ta, bool* out_tb) {
  bool ta = false, tb = false;

  const AttrBlob* ab = static_cast<const AttrBlob*>(attr);
  if (!ab) { *out_ta = ta; *out_tb = tb; return; }

  // accept schema_id==0 as "no attrs, defaults"
  if (ab->schema_id != 0 && ab->schema_id != kAttrSchema_Gemm) {
    *out_ta = ta; *out_tb = tb; return;
  }

  if (!ab->data || ab->bytes < (uint32_t)sizeof(GemmAttrV0)) {
    *out_ta = ta; *out_tb = tb; return;
  }

  GemmAttrV0 a{};
  std::memcpy(&a, ab->data, sizeof(GemmAttrV0));
  ta = (a.transA != 0);
  tb = (a.transB != 0);

  *out_ta = ta;
  *out_tb = tb;
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

  MatView2D A = make_view_2d(A0, transA); // [M,K]
  MatView2D B = make_view_2d(B0, transB); // [K,N]
  MatView2D C = make_view_2d(C0, false);  // [M,N]

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

// ---- WMMA helpers / pack / core ----
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
static bool gemm_f32_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  bool ta=false, tb=false;
  read_gemm_attr(attr, &ta, &tb);

  return gemm_check_2d(inputs, num_inputs, outputs, num_outputs,
                       ta, tb, DType::kF32, DType::kF32, DType::kF32);
}

static size_t gemm_f32_workspace(const TensorDesc*, int, const void*) { return 0; }

static Status gemm_f32_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  bool ta=false, tb=false;
  read_gemm_attr(attr, &ta, &tb);

  if (!gemm_check_2d(inputs, num_inputs, outputs, num_outputs,
                     ta, tb, DType::kF32, DType::kF32, DType::kF32)) {
    return Status::InvalidArgument;
  }

  MatView2D A = make_view_2d(inputs[0], ta);
  MatView2D B = make_view_2d(inputs[1], tb);
  MatView2D C = make_view_2d(outputs[0], false);

  const int M = (int)A.rows;
  const int K = (int)A.cols;
  const int N = (int)B.cols;

  dim3 block(16, 16, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y,
            1);

  cudaGetLastError(); // clear
  gemm_impl::gemm_f32_naive_strided_kernel<<<grid, block, 0, stream>>>(
      (const float*)A.data, A.rs, A.cs,
      (const float*)B.data, B.rs, B.cs,
      (float*)C.data, C.rs, C.cs,
      M, N, K);

  return cuda_last_status();
}

KernelVariant make_gemm_f32_naive_variant() {
  KernelVariant v{};
  v.name = "gemm_f32_naive_strided";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0; // accept schema 0 or GEMM (we parse both)
  v.launch = gemm_f32_launch;
  v.supported = gemm_f32_supported;
  v.query_workspace = gemm_f32_workspace;
  return v;
}

// ============================================================================
// f16 TC out_f16 variant
// ============================================================================
static bool gemm_f16_tc_out_f16_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  bool ta=false, tb=false;
  read_gemm_attr(attr, &ta, &tb);

  if (!gemm_check_2d(inputs, num_inputs, outputs, num_outputs,
                     ta, tb, DType::kF16, DType::kF16, DType::kF16)) {
    return false;
  }

  // TC path requires C contiguous row-major (as you noted)
  if (!is_contig_rowmajor_2d(outputs[0])) return false;

  return true;
}

static size_t gemm_f16_tc_out_f16_workspace(const TensorDesc*, int, const void*) { return 0; }

static Status gemm_f16_tc_out_f16_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void*, size_t,
    cudaStream_t stream) {

  bool ta=false, tb=false;
  read_gemm_attr(attr, &ta, &tb);

  if (!gemm_check_2d(inputs, num_inputs, outputs, num_outputs,
                     ta, tb, DType::kF16, DType::kF16, DType::kF16)) {
    return Status::InvalidArgument;
  }
  if (!is_contig_rowmajor_2d(outputs[0])) {
    return Status::InvalidArgument;
  }

  MatView2D A = make_view_2d(inputs[0], ta);
  MatView2D B = make_view_2d(inputs[1], tb);
  MatView2D C = make_view_2d(outputs[0], false);

  const int M = (int)A.rows;
  const int N = (int)B.cols;

  dim3 block(32, 1, 1);
  dim3 grid((N + 15) / 16, (M + 15) / 16, 1);

  cudaGetLastError(); // clear
  gemm_impl::gemm_f16_tc_wmma_out_f16_strided_kernel<<<grid, block, 0, stream>>>(
      (const __half*)A.data, A.rs, A.cs, A.rows, A.cols,
      (const __half*)B.data, B.rs, B.cs, B.rows, B.cols,
      (__half*)C.data, C.rs, C.cs, C.rows, C.cols);

  return cuda_last_status();
}

KernelVariant make_gemm_f16_tc_wmma_out_f16_variant() {
  KernelVariant v{};
  v.name = "gemm_f16_tc_wmma_out_f16_strided";
  v.priority = 20;
  v.flags = 0;
  v.expected_attr_schema_id = 0;
  v.launch = gemm_f16_tc_out_f16_launch;
  v.supported = gemm_f16_tc_out_f16_supported;
  v.query_workspace = gemm_f16_tc_out_f16_workspace;
  return v;
}

} // namespace aicf::cuda
