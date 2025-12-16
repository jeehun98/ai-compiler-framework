#pragma once

#include <aicf/core/status.hpp>
#include <aicf/ir/ir.hpp>
#include <aicf/runtime/graph.hpp>

namespace aicf {

// IR -> Runtime Graph 로 내리는 인터페이스 (현재는 더미)
Status lower_ir_to_runtime(const IRGraph& ir, Graph& out_graph);

} // namespace aicf
