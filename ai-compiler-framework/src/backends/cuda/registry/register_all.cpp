#include "aicf/backends/cuda/registry/register_all.hpp"

#include "aicf/backends/cuda/registry/registry.hpp"
#include "aicf/backends/cuda/registry/op_kind.hpp"
#include "aicf/backends/cuda/registry/kernel_variant.hpp"

namespace aicf::cuda {
KernelVariant make_add_f32_variant();
KernelVariant make_relu_f32_variant();
KernelVariant make_gemm_f32_naive_variant();

}

extern "C" void aicf_cuda_register_all_kernels() {
  using namespace aicf::cuda;

  auto& R = KernelRegistry::instance();
  R.register_kernel(OpKind::EltwiseAdd,  make_add_f32_variant());
  R.register_kernel(OpKind::EltwiseRelu, make_relu_f32_variant());
  R.register_kernel(OpKind::Gemm,        make_gemm_f32_naive_variant());
}
