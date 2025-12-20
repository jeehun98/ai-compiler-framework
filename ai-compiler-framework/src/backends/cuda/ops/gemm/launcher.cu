#include <cuda_runtime.h>

#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

// public API
#include <aicf/backends/cuda/ops/gemm/api.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

// common shim
#include "aicf/backends/cuda/ops/_common/shim/launch.hpp"
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"
#include "aicf/backends/cuda/ops/_common/shim/validate.hpp"

// optional (future): attrs helpers
// #include "aicf/backends/cuda/ops/_common/shim/attrs.hpp"

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
  for (int kk = 0; kk < K; ++kk) {
    acc += A[a_row_base + kk] * B[kk * N + col];
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
// Registry Variant - v0.2 Plan A (no workspace, no attr semantics yet)
//
// Contract:
//   inputs[0]=A [M,K], inputs[1]=B [K,N], outputs[0]=C [M,N]
//   binding guarantees: CUDA + contiguous (stride in desc is contiguous-by-contract)
//
// This variant supports:
//   - F32 only
// -------------------------

static inline bool gemm_variant_check_2d(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (num_inputs != 2 || num_outputs != 1) return false;

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  const TensorDesc& C = outputs[0];

  if (!aicf::cuda::shim::is_f32_contig_2d(A)) return false;
  if (!aicf::cuda::shim::is_f32_contig_2d(B)) return false;
  if (!aicf::cuda::shim::is_f32_contig_2d(C)) return false;

  if (!aicf::cuda::shim::gemm_shape_ok_2d(A, B, C)) return false;

  // extra guard
  if (A.shape[0] <= 0 || A.shape[1] <= 0 || B.shape[1] <= 0) return false;

  return true;
}

static bool gemm_variant_supported(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {

  if (!inputs || !outputs) return false;
  return gemm_variant_check_2d(inputs, num_inputs, outputs, num_outputs);
}

static size_t gemm_variant_workspace(const TensorDesc*, int, const void*) {
  return 0;
}

static aicf::Status gemm_variant_launch(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void* /*workspace*/, size_t /*workspace_bytes*/,
    cudaStream_t stream) {

  if (!inputs || !outputs) return aicf::Status::InvalidArgument;

  if (!gemm_variant_check_2d(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& A = inputs[0];
  const TensorDesc& B = inputs[1];
  TensorDesc& C = outputs[0];

  const int M = static_cast<int>(A.shape[0]);
  const int K = static_cast<int>(A.shape[1]);
  const int N = static_cast<int>(B.shape[1]);

  // Direct kernel launch on provided cudaStream_t.
  // (binding should pass PyTorch current stream)
  dim3 block(16, 16, 1);
  dim3 grid((N + block.x - 1) / block.x,
            (M + block.y - 1) / block.y,
            1);

  gemm_impl::gemm_f32_naive_kernel<<<grid, block, 0, stream>>>(
      (const float*)A.data,
      (const float*)B.data,
      (float*)C.data,
      M, N, K);

  return aicf::cuda::shim::cuda_last_error_to_status();

  // Alternative (keep public API path):
  // const aicf::Stream s = aicf::cuda::shim::from_cuda_stream(stream);
  // return aicf::cuda::gemm_f32((const float*)A.data, (const float*)B.data, (float*)C.data, M, N, K, s);
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
