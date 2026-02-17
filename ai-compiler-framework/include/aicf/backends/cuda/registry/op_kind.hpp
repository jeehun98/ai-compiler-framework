#pragma once

namespace aicf::cuda {

enum class OpKind : int {
  EltwiseAdd    = 0,
  EltwiseRelu   = 1,
  Gemm          = 2,
  BiasAdd       = 3,
  ReduceSum     = 4,
  MseGrad       = 5,
  ReluBwd       = 6,
  SgdStep       = 7,
  Copy          = 8,
  GradZero      = 9,
  AdamStep      = 10,
  StepInc       = 11,
  BiasCorr      = 12,

  LayerNormFwd  = 13,
  LayerNormBwd  = 14,

  BatchNormFwd  = 15,
  BatchNormBwd  = 16,

  GemmEpilogue  = 17,
  Softmax       = 18,
  SoftmaxBwd    = 19,

  MseLoss       = 20,

  // ---- NEW ----
  CrossEntropyFwd = 21,
  CrossEntropyBwd = 22,

  _Count        = 23
};

} // namespace aicf::cuda
