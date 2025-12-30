#pragma once

namespace aicf::cuda {

enum class OpKind : int {
  EltwiseAdd  = 0,
  EltwiseRelu = 1,
  Gemm        = 2,
  BiasAdd     = 3,
  ReduceSum   = 4,
  MseGrad     = 5,
  ReluBwd     = 6,
  SgdStep     = 7,
  Copy        = 8,
  GradZero    = 9,
  AdamStep    = 10,
  StepInc     = 11,
  BiasCorr    = 12,

  LayerNorm   = 13,

  _Count      = 14
};

}  // namespace aicf::cuda
