// ============================================================================
// src/backends/cuda/registry/register_all.cpp  (core-free / minimal)
// - registers only ReduceSum variants
// ============================================================================

#include "aicf/backends/cuda/registry/register_all.hpp"

#include "aicf/backends/cuda/registry/registry.hpp"
#include "aicf/backends/cuda/registry/op_kind.hpp"
#include "aicf/backends/cuda/registry/kernel_variant.hpp"

namespace aicf::cuda {

// Factories (must match actual launcher implementations)
KernelVariant make_reduce_sum_lastdim_f32_variant();
KernelVariant make_reduce_sum_lastdim_f16_to_f32_variant();

}  // namespace aicf::cuda

extern "C" void aicf_cuda_register_all_kernels() {
  using namespace aicf::cuda;

  auto& R = KernelRegistry::instance();

  auto setp = [](KernelVariant v, int p) {
    v.priority = p;
    return v;
  };

  // ReduceSum only
  {
    R.register_kernel(OpKind::ReduceSum, setp(make_reduce_sum_lastdim_f16_to_f32_variant(), 110));
    R.register_kernel(OpKind::ReduceSum, setp(make_reduce_sum_lastdim_f32_variant(), 100));
  }
}
