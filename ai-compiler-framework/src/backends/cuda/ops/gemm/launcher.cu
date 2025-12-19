#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/gemm/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

#include "kernels.cuh"

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
  // B is row-major [K,N]
  for (int kk = 0; kk < K; ++kk) {
    acc += A[a_row_base + kk] * B[kk * N + col];
  }
  C[row * N + col] = acc;
}

} // namespace gemm_impl

// -------------------------
// helpers
// -------------------------
static inline cudaStream_t to_cuda_stream(aicf::Stream s) {
  return (s.handle == nullptr) ? (cudaStream_t)0 : (cudaStream_t)s.handle;
}

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

  cudaStream_t s = to_cuda_stream(stream);

  dim3 block(16, 16, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y,
            1);

  gemm_impl::gemm_f32_naive_kernel<<<grid, block, 0, s>>>(A, B, C, M, N, K);

  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? aicf::Status::Ok : aicf::Status::Error;
}

// -------------------------
// Registry Variant (no attr) - v0.1 Plan A
// -------------------------
static aicf::Status gemm_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (num_inputs != 2 || num_outputs != 1) return aicf::Status::InvalidArgument;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& C = outputs[0];

  if (A.dtype != DType::kF32 || B.dtype != DType::kF32 || C.dtype != DType::kF32) {
    return aicf::Status::InvalidArgument;
  }

  // TensorDesc uses named union r.{rank,ndim}
  if (A.r.rank != 2 || B.r.rank != 2 || C.r.rank != 2) {
    return aicf::Status::InvalidArgument;
  }

  const int M  = (int)A.shape[0];
  const int K  = (int)A.shape[1];
  const int K2 = (int)B.shape[0];
  const int N  = (int)B.shape[1];

  if (M <= 0 || N <= 0 || K <= 0) return aicf::Status::InvalidArgument;
  if (K2 != K) return aicf::Status::InvalidArgument;
  if ((int)C.shape[0] != M || (int)C.shape[1] != N) return aicf::Status::InvalidArgument;

  aicf::Stream s{};
  s.handle = (void*)stream;

  return aicf::cuda::gemm_f32(
      (const float*)A.data,
      (const float*)B.data,
      (float*)C.data,
      M, N, K,
      s);
}

static bool gemm_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& C = outputs[0];

  if (A.dtype != DType::kF32 || B.dtype != DType::kF32 || C.dtype != DType::kF32) return false;
  if (A.r.rank != 2 || B.r.rank != 2 || C.r.rank != 2) return false;

  const int64_t M = A.shape[0];
  const int64_t K = A.shape[1];
  if (M <= 0 || K <= 0) return false;

  if (B.shape[0] != K) return false;
  const int64_t N = B.shape[1];
  if (N <= 0) return false;

  if (C.shape[0] != M || C.shape[1] != N) return false;

  return true;
}

static size_t gemm_variant_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

KernelVariant make_gemm_f32_naive_variant() {
  KernelVariant v;
  v.name = "gemm_f32_naive";
  v.launch = gemm_variant_launch;
  v.supported = gemm_variant_supported;
  v.query_workspace = gemm_variant_workspace;
  return v;
}

} // namespace aicf::cuda
