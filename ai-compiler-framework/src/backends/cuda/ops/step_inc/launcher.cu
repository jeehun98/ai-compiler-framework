#include <cuda_runtime.h>
#include <cstdint>

#include <aicf/core/status.hpp>

// registry glue
#include <aicf/backends/cuda/registry/kernel_variant.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>

// common shim
#include "aicf/backends/cuda/ops/_common/shim/status.hpp"

#include "kernels.cuh"

namespace aicf::cuda {

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

// step[i] += 1 (supports scalar: numel==1)
__global__ void step_inc_i32_kernel(int32_t* __restrict__ step, int64_t numel) {
  for (int64_t i = global_tid_1d(); i < numel; i += global_stride_1d()) {
    step[i] += 1;
  }
}

} // namespace step_inc_impl

// -------------------------
// Shape helpers
// -------------------------

// NOTE: step_inc는 scalar(rank=0)도 허용해야 함.
// TensorDesc가 rank==0이면 numel을 1로 취급.
static inline bool compute_numel_allow_scalar0d(const TensorDesc& T, int64_t* out) {
  if (!out) return false;

  const int64_t r = T.rank();
  if (r < 0) return false;

  if (r == 0) {  // 0-d scalar
    *out = 1;
    return true;
  }

  if (r > (int64_t)kMaxRank) return false;

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
  if (r == 0) return true; // both scalar
  for (int64_t i = 0; i < r; ++i) {
    if (A.shape[i] != B.shape[i]) return false;
  }
  return true;
}

static inline bool is_i32_contig_allow_scalar0d(const TensorDesc& T) {
  // scalar(0d)는 contiguous 플래그가 false로 올 수도 있어 애매함.
  // 하지만 step_inc는 1 element만 쓰기 때문에, rank==0이면 contiguous 검사 완화.
  if (T.dtype != DType::kI32) return false;
  if (T.rank() == 0) return true;
  return T.contiguous;
}

static inline int choose_blocks_1d(int64_t numel, int threads) {
  int64_t blocks64 = (numel + threads - 1) / threads;
  const int kMaxBlocks = 65535;
  int blocks = (blocks64 > (int64_t)kMaxBlocks) ? kMaxBlocks : (int)blocks64;
  if (blocks < 1) blocks = 1;
  return blocks;
}

static size_t step_inc_workspace(const TensorDesc*, int, const void*) { return 0; }

// ============================================================================
// Variant: StepInc i32
// Contract: inputs=(S), outputs=(Sout)
//   Sout = S + 1
// In-place allowed: outputs[0] may alias inputs[0] (and typical).
// ============================================================================
static inline bool step_inc_check_i32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs) {

  if (!inputs || !outputs) return false;
  if (num_inputs != 1 || num_outputs != 1) return false;

  const TensorDesc& S  = inputs[0];
  const TensorDesc& SO = outputs[0];

  if (!is_i32_contig_allow_scalar0d(S) || !is_i32_contig_allow_scalar0d(SO)) return false;
  if (!same_shape_allow_scalar0d(S, SO)) return false;

  int64_t numel = 0;
  if (!compute_numel_allow_scalar0d(S, &numel)) return false;

  // step는 scalar를 기대하지만, 안전하게 numel>=1이면 허용
  return (numel >= 1);
}

static bool step_inc_supported_i32(
    const TensorDesc* inputs, int num_inputs,
    const TensorDesc* outputs, int num_outputs,
    const void* /*attr*/) {
  return step_inc_check_i32(inputs, num_inputs, outputs, num_outputs);
}

static aicf::Status step_inc_launch_i32(
    const TensorDesc* inputs, int num_inputs,
    TensorDesc* outputs, int num_outputs,
    const void* /*attr*/,
    void*, size_t,
    cudaStream_t stream) {

  if (!step_inc_check_i32(inputs, num_inputs, outputs, num_outputs)) {
    return aicf::Status::InvalidArgument;
  }

  const TensorDesc& S  = inputs[0];
  TensorDesc& SO = outputs[0];

  // step_inc는 in-place가 정상 케이스.
  // out이 다른 버퍼여도 안전함.
  int64_t numel = 0;
  (void)compute_numel_allow_scalar0d(S, &numel);

  constexpr int kThreads = 256;
  const int blocks = choose_blocks_1d(numel, kThreads);

  step_inc_impl::step_inc_i32_kernel<<<blocks, kThreads, 0, stream>>>(
      (int32_t*)SO.data, numel);

  return aicf::cuda::shim::cuda_last_error_to_status();
}

// ============================================================
// IMPORTANT: register_all.cpp가 찾는 정확한 심볼 이름
// ============================================================
KernelVariant make_step_inc_variant() {
  KernelVariant v{};
  v.name = "step_inc_i32";
  v.priority = 0;
  v.flags = 0;
  v.launch = step_inc_launch_i32;
  v.supported = step_inc_supported_i32;
  v.query_workspace = step_inc_workspace;
  return v;
}

} // namespace aicf::cuda
