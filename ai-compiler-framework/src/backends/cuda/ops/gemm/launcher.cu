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
  // gemm/api.hpp 주석에 "stream.handle == nullptr" 라고 되어 있으니 handle 가정
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

  // 16x16 tile
  dim3 block(16, 16, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y,
            1);

  gemm_impl::gemm_f32_naive_kernel<<<grid, block, 0, s>>>(A, B, C, M, N, K);

  const cudaError_t e = cudaGetLastError();
  return (e == cudaSuccess) ? aicf::Status::Ok : aicf::Status::Error;
}

// -------------------------
// Registry Variant (no attr)
// Contract:
//   inputs[0]=A [M,K], inputs[1]=B [K,N], outputs[0]=C [M,N]
//   contiguous + dtype F32 + rank2 only
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

  if (A.dtype != DType::F32 || B.dtype != DType::F32 || C.dtype != DType::F32) {
    return aicf::Status::InvalidArgument;
  }
  if (!A.contiguous || !B.contiguous || !C.contiguous) {
    return aicf::Status::InvalidArgument;
  }
  if (A.ndim != 2 || B.ndim != 2 || C.ndim != 2) {
    return aicf::Status::InvalidArgument;
  }

  const int M  = (int)A.shape[0];
  const int K  = (int)A.shape[1];
  const int K2 = (int)B.shape[0];
  const int N  = (int)B.shape[1];

  if (K2 != K) return aicf::Status::InvalidArgument;
  if ((int)C.shape[0] != M || (int)C.shape[1] != N) return aicf::Status::InvalidArgument;

  aicf::Stream s{};
  s.handle = (void*)stream;   // stream.hpp의*
