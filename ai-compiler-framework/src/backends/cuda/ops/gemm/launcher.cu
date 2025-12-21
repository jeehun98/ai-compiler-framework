#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/gemm/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>   // ✅ NEW: AttrPack

// common shim
#include "aicf/backends/cuda/ops/_common/shim/launch.hpp"
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"
#include "aicf/backends/cuda/ops/_common/shim/validate.hpp"

// optional (future): attrs helpers
// #include "aicf/backends/cuda/ops/_common/shim/attrs.hpp"

#include "kernels.cuh"

#include <string_view>   // ✅ NEW

namespace aicf::cuda {

// -------------------------
// kernels
// -------------------------
namespace gemm_impl {

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

} // namespace gemm_impl

// -------------------------
// public API implementation
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

// -------------------------
// Registry Variant - v0.3 Plan A (no workspace, minimal attr semantics)
//
// Contract (base):
//   - A: [M,K] contiguous f32
//   - C: [M,N] contiguous f32
//   - B:
//       * transB=false: [K,N] contiguous f32
//       * transB=true : [N,K] f32 with stride (view ok)
//
// Attr semantics (minimal):
//   - transB : bool (default false)
//
// Notes:
//   - This keeps A/C contig-only for simplicity.
//   - B is allowed to be non-contiguous only in transB=true path.
// -------------------------

static inline bool gemm_variant_check_2d_ex(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    bool transB) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& C = outputs[0];

  // A, C: keep strict contig (simple baseline)
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
    // B must be contiguous [K,N]
    if (!aicf::cuda::shim::is_f32_contig_2d(B)) return false;
    if (B.shape[0] != K || B.shape[1] != N) return false;
  } else {
    // B is stored as [N,K] with arbitrary stride
    if (B.shape[0] != N || B.shape[1] != K) return false;
    if (B.stride[0] <= 0 || B.stride[1] <= 0) return false;
    // optional: allow contiguous too (works)
    // no further restriction
  }

  return true;
}

static bool gemm_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* attr) {

  if (!inputs || !outputs) return false;
  const bool transB = attr_get_bool(attr, "transB", false);
  return gemm_variant_check_2d_ex(inputs, num_inputs, outputs, num_outputs, transB);
}

static size_t gemm_variant_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

static aicf::Status gemm_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* attr,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;

  const bool transB = attr_get_bool(attr, "transB", false);

  if (!gemm_variant_check_2d_ex(inputs, num_inputs, outputs, num_outputs, transB)) {
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
  v.priority = 0;   // future: tiled/tc variants > naive
  v.flags = 0;
  v.launch = gemm_variant_launch;
  v.supported = gemm_variant_supported;
  v.query_workspace = gemm_variant_workspace;
  return v;
}

} // namespace aicf::cuda
