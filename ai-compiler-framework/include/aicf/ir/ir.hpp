#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace aicf {

// 진짜 IR은 나중에: 지금은 “컴파일러 레이어가 존재한다”는 골격만
enum class OpKind : int32_t {
    Unknown = 0,
    Gemm = 1,
    Conv2d = 2,
    Elementwise = 3,
};

struct IRNode {
    OpKind op = OpKind::Unknown;
    std::string debug_name;
};

struct IRGraph {
    std::vector<IRNode> nodes;
};

} // namespace aicf
