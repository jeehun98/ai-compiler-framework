#include <cuda_runtime.h>
#include <cstdint>

#include <aicf/backends/cuda/registry/status.hpp>
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>

#include "kernels.cuh"

namespace aicf::cuda {

// -------------------------
// cuda error -> Status (core-free)
// -------------------------
static inline Status cuda_to_status(cudaError_t e) {
  return (e == cudaSuccess) ? Status::Ok : Status::Internal;
}
static inline Status cuda_last_status() {
  return cuda_to_status(cudaGetLastError());
}

// -------------------------
// kernels (definitions live here)
// -------------------------
namespace step_inc_impl {

static __forceinline__ __device__ int64_t global_tid_1d() {
  return (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
}
static __forceinline__ __device__ int64_t global_stride_1d() {
  return (int64_t)gridDim.x * (int64_t)blockDim.x;
}

__global__ void step_inc_i32_kernel(int32_t* __restrict__ step, int64_t numel) {
  for (int64_t i = global_tid_1d(); i < numel; i += global_stride_1d()) {
    step[i] += 1;
  }
}

} // namespace step_inc_impl

// -------------------------
// helpers
// -------------------------
static inline bool compute_numel_allow_scalar0d(const TensorDesc& T, int64_t* out) {
  if (!out) return false;
  const int64_t r = T.rank();
  if (r < 0) return false;
  if (r == 0) { *out = 1; return true; }

  int64_t n = 1;
  for (int64_t i = 0; i < r; ++i) {
    const int64_t d = T.shape[i];
    if (d <= 0) return false;
    n *= d;
  }
  *out = n;
  return true;
}

static inline bool same_shape_allow_scalar0d(const TensorDesc& A, const TensorDesc& B) {
  if (A.rank() != B.rank()) return false;
  const int64_t r = A.rank();
  if (r == 0) return true;
  for (int64_t i = 0; i < r; ++i) {
    if (A.shape[i] != B.shape[i]) return false;
  }
  return true;
}

static inline bool is_i32_ok_allow_scalar0d(const TensorDesc& T) {
  if (T.dtype != DType::kI32) return false;
  if (T.rank() == 0) return true;     // scalar: relax contig
  return T.contiguous;                // rank>0: require contiguous
}

static inline int choose_blocks_1d(int64_t n, int threads) {
  int64_t b = (n + threads - 1) / threads;
  if (b < 1) b = 1;
  const int64_t kMaxBlocks = 65535;
  if (b > kMaxBlocks) b = kMaxBlocks;
  return (int)b;
}

static size_t step_inc_workspace(const TensorDesc*, int, const void*) { return 0; }

// -------------------------
// Contract:
// inputs[0]=S (i32), outputs[0]=SO (i32)
// same shape, scalar(0d) allowed, contig required for rank>0
// in-place allowed (SO may alias S)
// -------------------------
static inline bool step_inc_check_i32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& S  = inputs[0];
  const TensorDesc& SO = outputs[0];

  if (!is_i32_ok_allow_scalar0d(S) || !is_i32_ok_allow_scalar0d(SO)) return false;
  if (!same_shape_allow_scalar0d(S, SO)) return false;

  int64_t n = 0;
  if (!compute_numel_allow_scalar0d(S, &n)) return false;
  return (n >= 1);
}

static bool step_inc_supported_i32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return step_inc_check_i32(inputs, num_inputs, outputs, num_outputs);
}

static Status step_inc_launch_i32(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void*, size_t,
    cudaStream_t stream) {

  if (!step_inc_check_i32(inputs, num_inputs, outputs, num_outputs)) {
    return Status::InvalidArgument;
  }

  const TensorDesc& S  = inputs[0];
  TensorDesc& SO       = outputs[0];

  int64_t n = 0;
  (void)compute_numel_allow_scalar0d(S, &n);
  if (n <= 0) return Status::Ok;

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(n, kThreads);

  cudaGetLastError(); // clear
  step_inc_impl::step_inc_i32_kernel<<<blocks, kThreads, 0, stream>>>(
      (int32_t*)SO.data, n);

  return cuda_last_status();
}

// IMPORTANT: register_all.cpp가 찾는 정확한 심볼
KernelVariant make_step_inc_variant() {
  KernelVariant v{};
  v.name = "step_inc_i32";
  v.priority = 0;
  v.flags = 0;
  v.expected_attr_schema_id = 0; // no attrs
  v.launch = step_inc_launch_i32;
  v.supported = step_inc_supported_i32;
  v.query_workspace = step_inc_workspace;
  return v;
}

} // namespace aicf::cuda
