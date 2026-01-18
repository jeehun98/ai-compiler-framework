// ============================================================================
// src/backends/cuda/registry/register_all.cpp  (core-free / minimal)
// - registers ReduceSum + Gemm variants
// ============================================================================

#include "aicf/backends/cuda/registry/register_all.hpp"

#include "aicf/backends/cuda/registry/registry.hpp"
#include "aicf/backends/cuda/registry/op_kind.hpp"
#include "aicf/backends/cuda/registry/kernel_variant.hpp"

namespace aicf::cuda {

// ReduceSum
KernelVariant make_reduce_sum_lastdim_f32_variant();
KernelVariant make_reduce_sum_lastdim_f16_to_f32_variant();

// Gemm
KernelVariant make_gemm_f32_naive_variant();
KernelVariant make_gemm_f16_tc_wmma_out_f16_variant();

}  // namespace aicf::cuda

extern "C" void aicf_cuda_register_all_kernels() {
  using namespace aicf::cuda;

  auto& R = KernelRegistry::instance();

  auto setp = [](KernelVariant v, int p) {
    v.priority = p;
    return v;
  };

  // ReduceSum
  {
    R.register_kernel(OpKind::ReduceSum, setp(make_reduce_sum_lastdim_f16_to_f32_variant(), 110));
    R.register_kernel(OpKind::ReduceSum, setp(make_reduce_sum_lastdim_f32_variant(), 100));
  }

  // Gemm
  {
    R.register_kernel(OpKind::Gemm, setp(make_gemm_f16_tc_wmma_out_f16_variant(), 20));
    R.register_kernel(OpKind::Gemm, setp(make_gemm_f32_naive_variant(), 0));
  }
}
