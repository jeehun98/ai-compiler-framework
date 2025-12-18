#pragma once

namespace aicf::cuda {

enum class OpKind : int {
  EltwiseAdd = 0,
  EltwiseRelu = 1,
  Gemm = 2,
  _Count = 3
};

}  // namespace aicf::cuda
