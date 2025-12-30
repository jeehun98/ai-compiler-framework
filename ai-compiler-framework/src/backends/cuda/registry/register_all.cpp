// ============================================================================
// src/backends/cuda/registry/register_all.cpp
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
KernelVariant make_add_f16_vec2_variant();

// EltwiseRelu
KernelVariant make_relu_f32_variant();
KernelVariant make_relu_f16_variant();
KernelVariant make_relu_f16_vec2_variant();

// Gemm
KernelVariant make_gemm_f32_naive_variant();
KernelVariant make_gemm_f16_tc_wmma_out_f16_variant();

// BiasAdd
KernelVariant make_bias_add_f32_variant();
KernelVariant make_bias_add_f16_variant();
KernelVariant make_bias_add_f16_vec2_variant();

// ReduceSum
KernelVariant make_reduce_sum_lastdim_f32_variant();
KernelVariant make_reduce_sum_lastdim_f16_to_f32_variant();

// MseGrad
KernelVariant make_mse_grad_f32_variant();
KernelVariant make_mse_grad_f16_variant();

// ReluBwd
KernelVariant make_relu_bwd_f32_variant();
KernelVariant make_relu_bwd_f16_variant();
KernelVariant make_relu_bwd_f16_vec2_variant();

// SgdStep
KernelVariant make_sgd_step_f32_variant();
KernelVariant make_sgd_step_f16_variant();
KernelVariant make_sgd_step_f16_half2_variant();

// Copy
KernelVariant make_copy_f32_variant();
KernelVariant make_copy_f16_variant();
KernelVariant make_copy_f16_vec2_variant();

// GradZero
KernelVariant make_grad_zero_variant();

// AdamStep
KernelVariant make_adam_step_f32_variant();

// StepInc
KernelVariant make_step_inc_variant();

// BiasCorr
KernelVariant make_biascorr_variant();

// ---- NEW: LayerNorm ----
KernelVariant make_layernorm_fwd_f16_variant();
KernelVariant make_layernorm_fwd_f32_variant();
KernelVariant make_layernorm_bwd_f16_variant();
KernelVariant make_layernorm_bwd_f32_variant();


// ---- NEW: BatchNorm ----
KernelVariant make_batchnorm_fwd_f16_variant();
KernelVariant make_batchnorm_bwd_f16_variant();


}  // namespace aicf::cuda

extern "C" void aicf_cuda_register_all_kernels() {
  using namespace aicf::cuda;
  auto& R = KernelRegistry::instance();

  // EltwiseAdd
  {
    R.register_kernel(OpKind::EltwiseAdd, make_add_f16_vec2_variant());
    R.register_kernel(OpKind::EltwiseAdd, make_add_f16_variant());
    R.register_kernel(OpKind::EltwiseAdd, make_add_f32_variant());
  }

  // EltwiseRelu
  {
    R.register_kernel(OpKind::EltwiseRelu, make_relu_f16_vec2_variant());
    R.register_kernel(OpKind::EltwiseRelu, make_relu_f16_variant());
    R.register_kernel(OpKind::EltwiseRelu, make_relu_f32_variant());
  }

  // Gemm
  {
    R.register_kernel(OpKind::Gemm, make_gemm_f16_tc_wmma_out_f16_variant());
    R.register_kernel(OpKind::Gemm, make_gemm_f32_naive_variant());
  }

  // BiasAdd
  {
    R.register_kernel(OpKind::BiasAdd, make_bias_add_f16_vec2_variant());
    R.register_kernel(OpKind::BiasAdd, make_bias_add_f16_variant());
    R.register_kernel(OpKind::BiasAdd, make_bias_add_f32_variant());
  }

  // ReduceSum
  {
    R.register_kernel(OpKind::ReduceSum, make_reduce_sum_lastdim_f16_to_f32_variant());
    R.register_kernel(OpKind::ReduceSum, make_reduce_sum_lastdim_f32_variant());
  }

  // MseGrad
  {
    R.register_kernel(OpKind::MseGrad, make_mse_grad_f16_variant());
    R.register_kernel(OpKind::MseGrad, make_mse_grad_f32_variant());
  }

  // ReluBwd
  {
    R.register_kernel(OpKind::ReluBwd, make_relu_bwd_f16_vec2_variant());
    R.register_kernel(OpKind::ReluBwd, make_relu_bwd_f16_variant());
    R.register_kernel(OpKind::ReluBwd, make_relu_bwd_f32_variant());
  }

  // SgdStep
  {
    R.register_kernel(OpKind::SgdStep, make_sgd_step_f16_half2_variant());
    R.register_kernel(OpKind::SgdStep, make_sgd_step_f16_variant());
    R.register_kernel(OpKind::SgdStep, make_sgd_step_f32_variant());
  }

  // Copy
  {
    R.register_kernel(OpKind::Copy, make_copy_f16_vec2_variant());
    R.register_kernel(OpKind::Copy, make_copy_f16_variant());
    R.register_kernel(OpKind::Copy, make_copy_f32_variant());
  }

  // GradZero
  { R.register_kernel(OpKind::GradZero, make_grad_zero_variant()); }

  // AdamStep
  { R.register_kernel(OpKind::AdamStep, make_adam_step_f32_variant()); }

  // StepInc
  { R.register_kernel(OpKind::StepInc, make_step_inc_variant()); }

  // BiasCorr
  { R.register_kernel(OpKind::BiasCorr, make_biascorr_variant()); }

  // ---- NEW: LayerNormFwd ----
  {
    R.register_kernel(OpKind::LayerNormFwd, make_layernorm_fwd_f16_variant());
    R.register_kernel(OpKind::LayerNormFwd, make_layernorm_fwd_f32_variant());
  }

  // ---- NEW: LayerNormBwd ----
  {
    R.register_kernel(OpKind::LayerNormBwd, make_layernorm_bwd_f16_variant());
    R.register_kernel(OpKind::LayerNormBwd, make_layernorm_bwd_f32_variant());
  }

    // ---- NEW: BatchNormFwd ----
  {
    R.register_kernel(OpKind::BatchNormFwd, make_batchnorm_fwd_f16_variant());
  }

  // ---- NEW: BatchNormBwd ----
  {
    R.register_kernel(OpKind::BatchNormBwd, make_batchnorm_bwd_f16_variant());
  }

}
