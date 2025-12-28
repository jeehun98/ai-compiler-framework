// ============================================================================
// src/backends/cuda/registry/register_all.cpp
// - Register kernels based on current launcher-implemented variants
// - Preference order: vec2/half2 > f16 scalar > f32 (when same op + valid supported())
// ============================================================================

#include "aicf/backends/cuda/registry/register_all.hpp"

#include "aicf/backends/cuda/registry/registry.hpp"
#include "aicf/backends/cuda/registry/op_kind.hpp"
#include "aicf/backends/cuda/registry/kernel_variant.hpp"

namespace aicf::cuda {

// -------------------------
// Factories (must match actual launcher implementations)
// -------------------------

// EltwiseAdd
KernelVariant make_add_f32_variant();
KernelVariant make_add_f16_variant();
KernelVariant make_add_f16_vec2_variant();   // half2

// EltwiseRelu
KernelVariant make_relu_f32_variant();
KernelVariant make_relu_f16_variant();
KernelVariant make_relu_f16_vec2_variant();  // half2

// Gemm
KernelVariant make_gemm_f32_naive_variant();
// [UPDATED] unified TC f16 in -> f16 out (handles transA/transB in supported())
KernelVariant make_gemm_f16_tc_wmma_out_f16_variant();

// BiasAdd
KernelVariant make_bias_add_f32_variant();
KernelVariant make_bias_add_f16_variant();
KernelVariant make_bias_add_f16_vec2_variant(); // half2

// ReduceSum
KernelVariant make_reduce_sum_lastdim_f32_variant();
KernelVariant make_reduce_sum_lastdim_f16_to_f32_variant();

// MseGrad
KernelVariant make_mse_grad_f32_variant();
KernelVariant make_mse_grad_f16_variant();

// ReluBwd
KernelVariant make_relu_bwd_f32_variant();
KernelVariant make_relu_bwd_f16_variant();
KernelVariant make_relu_bwd_f16_vec2_variant(); // half2

// SgdStep
KernelVariant make_sgd_step_f32_variant();
KernelVariant make_sgd_step_f16_variant();
KernelVariant make_sgd_step_f16_half2_variant();

// Copy
KernelVariant make_copy_f32_variant();
KernelVariant make_copy_f16_variant();
KernelVariant make_copy_f16_vec2_variant(); // half2



}  // namespace aicf::cuda

extern "C" void aicf_cuda_register_all_kernels() {
  using namespace aicf::cuda;

  auto& R = KernelRegistry::instance();

  // --------------------------------------------------------------------------
  // EltwiseAdd
  // --------------------------------------------------------------------------
  {
    R.register_kernel(OpKind::EltwiseAdd, make_add_f16_vec2_variant());
    R.register_kernel(OpKind::EltwiseAdd, make_add_f16_variant());
    R.register_kernel(OpKind::EltwiseAdd, make_add_f32_variant());
  }

  // --------------------------------------------------------------------------
  // EltwiseRelu
  // --------------------------------------------------------------------------
  {
    R.register_kernel(OpKind::EltwiseRelu, make_relu_f16_vec2_variant());
    R.register_kernel(OpKind::EltwiseRelu, make_relu_f16_variant());
    R.register_kernel(OpKind::EltwiseRelu, make_relu_f32_variant());
  }

  // --------------------------------------------------------------------------
  // Gemm
  // --------------------------------------------------------------------------
  {
    // Prefer unified TC variant (it internally gates by dtype/trans/stride)
    R.register_kernel(OpKind::Gemm, make_gemm_f16_tc_wmma_out_f16_variant());

    // Fallback / baseline
    R.register_kernel(OpKind::Gemm, make_gemm_f32_naive_variant());
  }

  // --------------------------------------------------------------------------
  // BiasAdd
  // --------------------------------------------------------------------------
  {
    R.register_kernel(OpKind::BiasAdd, make_bias_add_f16_vec2_variant());
    R.register_kernel(OpKind::BiasAdd, make_bias_add_f16_variant());
    R.register_kernel(OpKind::BiasAdd, make_bias_add_f32_variant());
  }

  // --------------------------------------------------------------------------
  // ReduceSum
  // --------------------------------------------------------------------------
  {
    // Specialized: f16 input -> f32 output (if implemented)
    R.register_kernel(OpKind::ReduceSum, make_reduce_sum_lastdim_f16_to_f32_variant());
    // Fallback / baseline
    R.register_kernel(OpKind::ReduceSum, make_reduce_sum_lastdim_f32_variant());
  }

  // --------------------------------------------------------------------------
  // MseGrad
  // --------------------------------------------------------------------------
  {
    R.register_kernel(OpKind::MseGrad, make_mse_grad_f16_variant());
    R.register_kernel(OpKind::MseGrad, make_mse_grad_f32_variant());
  }

  // --------------------------------------------------------------------------
  // ReluBwd
  // --------------------------------------------------------------------------
  {
    R.register_kernel(OpKind::ReluBwd, make_relu_bwd_f16_vec2_variant());
    R.register_kernel(OpKind::ReluBwd, make_relu_bwd_f16_variant());
    R.register_kernel(OpKind::ReluBwd, make_relu_bwd_f32_variant());
  }

  // --------------------------------------------------------------------------
  // SgdStep
  // --------------------------------------------------------------------------
  {
    R.register_kernel(OpKind::SgdStep, make_sgd_step_f16_half2_variant());
    R.register_kernel(OpKind::SgdStep, make_sgd_step_f16_variant());
    R.register_kernel(OpKind::SgdStep, make_sgd_step_f32_variant());
  }
  
  // --------------------------------------------------------------------------
  // Copy
  // --------------------------------------------------------------------------
  {
    R.register_kernel(OpKind::Copy, make_copy_f16_vec2_variant());
    R.register_kernel(OpKind::Copy, make_copy_f16_variant());
    R.register_kernel(OpKind::Copy, make_copy_f32_variant());
  }

}
