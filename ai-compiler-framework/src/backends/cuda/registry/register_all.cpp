#include "aicf/backends/cuda/registry/register_all.hpp"

#include "aicf/backends/cuda/registry/registry.hpp"
#include "aicf/backends/cuda/registry/op_kind.hpp"
#include "aicf/backends/cuda/registry/kernel_variant.hpp"

namespace aicf::cuda {

// Factories (v0.2): set name/priority inside each factory.
KernelVariant make_add_f32_variant();
KernelVariant make_relu_f32_variant();
KernelVariant make_gemm_f32_naive_variant();

// Future placeholders (optional)
KernelVariant make_add_f16_variant();
KernelVariant make_relu_f16_variant();
// KernelVariant make_gemm_f16_variant();

KernelVariant make_add_f16_vec2_variant();

}  // namespace aicf::cuda

extern "C" void aicf_cuda_register_all_kernels() {
  using namespace aicf::cuda;

  auto& R = KernelRegistry::instance();

  // EltwiseAdd
  {
    auto v = make_add_f32_variant();     // v.priority can be set in factory
    R.register_kernel(OpKind::EltwiseAdd, v);
    
    auto v16 = make_add_f16_variant();
    R.register_kernel(OpKind::EltwiseAdd, v16);
  }

  // EltwiseRelu
  {
    auto v = make_relu_f32_variant();
    R.register_kernel(OpKind::EltwiseRelu, v);
    
    auto v16 = make_relu_f16_variant();
    R.register_kernel(OpKind::EltwiseRelu, make_relu_f16_variant());

    R.register_kernel(OpKind::EltwiseAdd, make_add_f16_vec2_variant());   // half2 (priority=10)

  }

  // Gemm
  {
    auto v = make_gemm_f32_naive_variant();
    R.register_kernel(OpKind::Gemm, v);
    // R.register_kernel(OpKind::Gemm, make_gemm_f16_variant());
  }
}
