#pragma once

namespace aicf::cuda {

enum class OpKind : int {
  EltwiseAdd = 0,
  EltwiseRelu = 1,
  Gemm = 2,
  BiasAdd = 3,
  _Count = 4
};

}  // namespace aicf::cuda
