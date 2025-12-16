#include <aicf/runtime/graph.hpp>
#include <aicf/core/log.hpp>

namespace aicf {

Status init_runtime() {
    log_info("aicf runtime init()");
    return Status::Ok;
}

Status shutdown_runtime() {
    log_info("aicf runtime shutdown()");
    return Status::Ok;
}

} // namespace aicf
