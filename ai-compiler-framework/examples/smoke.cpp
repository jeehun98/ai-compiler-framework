#include <aicf/runtime/graph.hpp>
#include <aicf/core/log.hpp>

int main() {
    aicf::init_runtime();
    aicf::log_info("smoke ok");
    aicf::shutdown_runtime();
    return 0;
}
