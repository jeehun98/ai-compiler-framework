#include "aicf/backends/cuda/registry/register_all.hpp"

#include "aicf/backends/cuda/registry/registry.hpp"
#include "aicf/backends/cuda/registry/op_kind.hpp"
#include "aicf/backends/cuda/registry/kernel_variant.hpp"

namespace aicf::cuda {

// Factories
KernelVariant make_add_f32_variant();
KernelVariant make_relu_f32_variant();

KernelVariant make_gemm_f32_naive_variant();
KernelVariant make_gemm_f16_tc_wmma_variant();

KernelVariant make_bias_add_f32_variant();
KernelVariant make_reduce_sum_lastdim_f32_variant();
KernelVariant make_mse_grad_f32_variant();

// ✅ NEW: ReLU backward
KernelVariant make_relu_bwd_f32_variant();

// Future placeholders
KernelVariant make_add_f16_variant();
KernelVariant make_relu_f16_variant();
KernelVariant make_add_f16_vec2_variant();

}  // namespace aicf::cuda

extern "C" void aicf_cuda_register_all_kernels() {
  using namespace aicf::cuda;

  auto& R = KernelRegistry::instance();

  // EltwiseAdd
  {
    R.register_kernel(OpKind::EltwiseAdd, make_add_f32_variant());
    R.register_kernel(OpKind::EltwiseAdd, make_add_f16_variant());
    R.register_kernel(OpKind::EltwiseAdd, make_add_f16_vec2_variant());
  }

  // EltwiseRelu
  {
    R.register_kernel(OpKind::EltwiseRelu, make_relu_f32_variant());
    R.register_kernel(OpKind::EltwiseRelu, make_relu_f16_variant());
  }

  // Gemm
  {
    R.register_kernel(OpKind::Gemm, make_gemm_f16_tc_wmma_variant());
    R.register_kernel(OpKind::Gemm, make_gemm_f32_naive_variant());
  }

  // BiasAdd
  {
    R.register_kernel(OpKind::BiasAdd, make_bias_add_f32_variant());
  }

  // ReduceSum
  {
    R.register_kernel(OpKind::ReduceSum, make_reduce_sum_lastdim_f32_variant());
  }

  // MseGrad
  {
    R.register_kernel(OpKind::MseGrad, make_mse_grad_f32_variant());
  }

  // ✅ ReLU backward
  {
    R.register_kernel(OpKind::ReluBwd, make_relu_bwd_f32_variant());
  }
}
