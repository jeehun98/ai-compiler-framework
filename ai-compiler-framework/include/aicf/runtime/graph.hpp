#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <aicf/core/status.hpp>

namespace aicf {

// 지금은 “골격”만: 나중에 GraphKey/Plan/Exec로 커질 자리
struct Graph {
    std::string name;
};

struct CapturePlan {
    std::string name;
};

struct GraphExec {
    std::string name;
};

Status init_runtime();
Status shutdown_runtime();

} // namespace aicf
