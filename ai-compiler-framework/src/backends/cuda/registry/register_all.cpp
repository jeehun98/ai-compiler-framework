// ============================================================================
// src/backends/cuda/registry/register_all.cpp  (core-free / minimal)
// - registers selected CUDA kernel variants into KernelRegistry
// - 결정 박제(compiler) 지원을 위해 KernelVariant.kernel_id를 강제 세팅한다.
// ============================================================================

#include "aicf/backends/cuda/registry/register_all.hpp"

#include "aicf/backends/cuda/registry/registry.hpp"
#include "aicf/backends/cuda/registry/op_kind.hpp"
#include "aicf/backends/cuda/registry/kernel_variant.hpp"

namespace aicf::cuda {

// ReduceSum
KernelVariant make_reduce_sum_lastdim_f32_variant();
KernelVariant make_reduce_sum_lastdim_f16_to_f32_variant();
KernelVariant make_reduce_sum_lastdim_f16_variant();   // ✅ ADD

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

// LayerNormFwd
KernelVariant make_layernorm_fwd_f32_variant();
KernelVariant make_layernorm_fwd_f16_variant();

// LayerNormBwd
KernelVariant make_layernorm_bwd_f32_variant();
KernelVariant make_layernorm_bwd_f16_variant();

// BatchNorm
KernelVariant make_batchnorm_fwd_f16_variant();
KernelVariant make_batchnorm_bwd_f16_variant();

}  // namespace aicf::cuda

extern "C" void aicf_cuda_register_all_kernels() {
  using namespace aicf::cuda;

  std::fprintf(stderr, "[aicf] register_all: KID_VERSION=2026-01-20\n");
  auto& R = KernelRegistry::instance();

  auto setp = [](KernelVariant v, int p) {
    v.priority = p;
    return v;
  };

  // 결정 박제용: kernel_id를 반드시 채운다.
  auto kid = [](KernelVariant v, const char* id) {
    v.kernel_id = id;  // KernelVariant에 kernel_id 필드가 있어야 함
    return v;
  };

  // ReduceSum
  {
    // ✅ f16 -> f16
    R.register_kernel(OpKind::ReduceSum,
      kid(setp(make_reduce_sum_lastdim_f16_variant(), 115),
          "reduce_sum_keep_lastdim_f16_v0"));

    // f16 -> f32
    R.register_kernel(OpKind::ReduceSum,
      kid(setp(make_reduce_sum_lastdim_f16_to_f32_variant(), 110),
          "reduce_sum_keep_lastdim_f16_to_f32_v0"));

    // f32 -> f32
    R.register_kernel(OpKind::ReduceSum,
      kid(setp(make_reduce_sum_lastdim_f32_variant(), 100),
          "reduce_sum_keep_lastdim_f32_v0"));
  }



  // Gemm
  {
    R.register_kernel(OpKind::Gemm,
      kid(setp(make_gemm_f16_tc_wmma_out_f16_variant(), 20),
          "gemm_f16_tc_wmma_out_f16_v0"));
    R.register_kernel(OpKind::Gemm,
      kid(setp(make_gemm_f32_naive_variant(), 0),
          "gemm_f32_naive_v0"));
  }

  // BiasAdd
  {
    R.register_kernel(OpKind::BiasAdd,
      kid(setp(make_bias_add_f16_vec2_variant(), 30),
          "bias_add_f16_vec2_v0"));
    R.register_kernel(OpKind::BiasAdd,
      kid(setp(make_bias_add_f16_variant(), 20),
          "bias_add_f16_v0"));
    R.register_kernel(OpKind::BiasAdd,
      kid(setp(make_bias_add_f32_variant(), 0),
          "bias_add_f32_v0"));
  }

  // Add
  {
    R.register_kernel(OpKind::EltwiseAdd,
      kid(setp(make_add_f16_vec2_variant(), 30),
          "add_f16_vec2_v0"));
    R.register_kernel(OpKind::EltwiseAdd,
      kid(setp(make_add_f16_variant(), 20),
          "add_f16_v0"));
    R.register_kernel(OpKind::EltwiseAdd,
      kid(setp(make_add_f32_variant(), 0),
          "add_f32_v0"));
  }

  // Relu
  {
    R.register_kernel(OpKind::EltwiseRelu,
      kid(setp(make_relu_f16_vec2_variant(), 30),
          "relu_f16_vec2_v0"));
    R.register_kernel(OpKind::EltwiseRelu,
      kid(setp(make_relu_f16_variant(), 20),
          "relu_f16_v0"));
    R.register_kernel(OpKind::EltwiseRelu,
      kid(setp(make_relu_f32_variant(), 0),
          "relu_f32_v0"));
  }

  // MseGrad
  {
    R.register_kernel(OpKind::MseGrad,
      kid(setp(make_mse_grad_f16_variant(), 20),
          "mse_grad_f16_v0"));
    R.register_kernel(OpKind::MseGrad,
      kid(setp(make_mse_grad_f32_variant(), 0),
          "mse_grad_f32_v0"));
  }

  // ReluBwd
  {
    R.register_kernel(OpKind::ReluBwd,
      kid(setp(make_relu_bwd_f16_vec2_variant(), 30),
          "relu_bwd_f16_vec2_v0"));
    R.register_kernel(OpKind::ReluBwd,
      kid(setp(make_relu_bwd_f16_variant(), 20),
          "relu_bwd_f16_v0"));
    R.register_kernel(OpKind::ReluBwd,
      kid(setp(make_relu_bwd_f32_variant(), 0),
          "relu_bwd_f32_v0"));
  }

  // SgdStep
  {
    R.register_kernel(OpKind::SgdStep,
      kid(setp(make_sgd_step_f16_half2_variant(), 30),
          "sgd_step_f16_half2_v0"));
    R.register_kernel(OpKind::SgdStep,
      kid(setp(make_sgd_step_f16_variant(), 20),
          "sgd_step_f16_v0"));
    R.register_kernel(OpKind::SgdStep,
      kid(setp(make_sgd_step_f32_variant(), 0),
          "sgd_step_f32_v0"));
  }

  // Copy
  {
    R.register_kernel(OpKind::Copy,
      kid(setp(make_copy_f16_variant(), 20),
          "copy_f16_v0"));
    R.register_kernel(OpKind::Copy,
      kid(setp(make_copy_f32_variant(), 0),
          "copy_f32_v0"));
  }

  // GradZero
  {
    R.register_kernel(OpKind::GradZero,
      kid(setp(make_grad_zero_variant(), 0),
          "grad_zero_v0"));
  }

  // AdamStep
  {
    R.register_kernel(OpKind::AdamStep,
      kid(setp(make_adam_step_f32_variant(), 0),
          "adam_step_f32_v0"));
  }

  // StepInc
  {
    R.register_kernel(OpKind::StepInc,
      kid(setp(make_step_inc_variant(), 0),
          "step_inc_v0"));
  }

  // BiasCorr
  {
    R.register_kernel(OpKind::BiasCorr,
      kid(setp(make_biascorr_variant(), 0),
          "bias_corr_v0"));
  }

  // LayerNormFwd
  {
    R.register_kernel(OpKind::LayerNormFwd,
      kid(setp(make_layernorm_fwd_f16_variant(), 10),
          "layernorm_fwd_f16_v0"));
    R.register_kernel(OpKind::LayerNormFwd,
      kid(setp(make_layernorm_fwd_f32_variant(), 0),
          "layernorm_fwd_f32_v0"));
  }

  // LayerNormBwd
  {
    R.register_kernel(OpKind::LayerNormBwd,
      kid(setp(make_layernorm_bwd_f16_variant(), 10),
          "layernorm_bwd_f16_v0"));
    R.register_kernel(OpKind::LayerNormBwd,
      kid(setp(make_layernorm_bwd_f32_variant(), 0),
          "layernorm_bwd_f32_v0"));
  }

  // BatchNormFwd
  {
    R.register_kernel(OpKind::BatchNormFwd,
      kid(setp(make_batchnorm_fwd_f16_variant(), 10),
          "batchnorm_fwd_f16_v0"));
  }

  // BatchNormBwd
  {
    R.register_kernel(OpKind::BatchNormBwd,
      kid(setp(make_batchnorm_bwd_f16_variant(), 10),
          "batchnorm_bwd_f16_v0"));
  }
}
