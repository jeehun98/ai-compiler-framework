#include <aicf/ir/lowering.hpp>
#include <aicf/core/log.hpp>

namespace aicf {

Status lower_ir_to_runtime(const IRGraph& ir, Graph& out_graph) {
    out_graph.name = "lowered_graph";
    log_info("lower_ir_to_runtime(): nodes=%zu", ir.nodes.size());
    return Status::Ok;
}

} // namespace aicf
