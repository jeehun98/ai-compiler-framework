// ============================================================================
// src/backends/cuda/registry/register_all.cpp  (core-free / minimal)
// - registers selected CUDA kernel variants into KernelRegistry
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

// BiasAdd
KernelVariant make_bias_add_f32_variant();
KernelVariant make_bias_add_f16_variant();
KernelVariant make_bias_add_f16_vec2_variant();

// Add
KernelVariant make_add_f32_variant();
KernelVariant make_add_f16_variant();
KernelVariant make_add_f16_vec2_variant();

// ReLu
KernelVariant make_relu_f32_variant();
KernelVariant make_relu_f16_variant();
KernelVariant make_relu_f16_vec2_variant();

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

// GradZero
KernelVariant make_grad_zero_variant();

// AdamStep
KernelVariant make_adam_step_f32_variant();

// StepInc
KernelVariant make_step_inc_variant();

// BiasCorr
KernelVariant make_biascorr_variant();
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

  // BiasAdd
  {
    R.register_kernel(OpKind::BiasAdd, setp(make_bias_add_f16_vec2_variant(), 30));
    R.register_kernel(OpKind::BiasAdd, setp(make_bias_add_f16_variant(), 20));
    R.register_kernel(OpKind::BiasAdd, setp(make_bias_add_f32_variant(), 0));
  }

  // Add
  {
    R.register_kernel(OpKind::EltwiseAdd, setp(make_add_f16_vec2_variant(), 30));
    R.register_kernel(OpKind::EltwiseAdd, setp(make_add_f16_variant(), 20));
    R.register_kernel(OpKind::EltwiseAdd, setp(make_add_f32_variant(), 0));
  }

  // Relu
  {
    R.register_kernel(OpKind::EltwiseRelu, setp(make_relu_f16_vec2_variant(), 30));
    R.register_kernel(OpKind::EltwiseRelu, setp(make_relu_f16_variant(), 20));
    R.register_kernel(OpKind::EltwiseRelu, setp(make_relu_f32_variant(), 0));
  }

  // MseGrad
  {
    R.register_kernel(OpKind::MseGrad, setp(make_mse_grad_f16_variant(), 20));
    R.register_kernel(OpKind::MseGrad, setp(make_mse_grad_f32_variant(), 0));
  }

  // ReluBwd
  {
    R.register_kernel(OpKind::ReluBwd, setp(make_relu_bwd_f16_vec2_variant(), 30));
    R.register_kernel(OpKind::ReluBwd, setp(make_relu_bwd_f16_variant(), 20));
    R.register_kernel(OpKind::ReluBwd, setp(make_relu_bwd_f32_variant(), 0));
  }

  // SgdStep
  {
    R.register_kernel(OpKind::SgdStep, setp(make_sgd_step_f16_half2_variant(), 30));
    R.register_kernel(OpKind::SgdStep, setp(make_sgd_step_f16_variant(), 20));
    R.register_kernel(OpKind::SgdStep, setp(make_sgd_step_f32_variant(), 0));
  }

  // Copy
  {
    R.register_kernel(OpKind::Copy, setp(make_copy_f16_variant(), 20));
    R.register_kernel(OpKind::Copy, setp(make_copy_f32_variant(), 0));
  }

  // GradZero
  {
    R.register_kernel(OpKind::GradZero, setp(make_grad_zero_variant(), 0));
  }

  // AdamStep
  {
    R.register_kernel(OpKind::AdamStep, setp(make_adam_step_f32_variant(), 0));
  }

  // StepInc
  {
    R.register_kernel(OpKind::StepInc, setp(make_step_inc_variant(), 0));
  }

  // BiasCorr
  {
    R.register_kernel(OpKind::BiasCorr, setp(make_biascorr_variant(), 0));
  }

}
