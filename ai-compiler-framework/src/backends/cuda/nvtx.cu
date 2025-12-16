#include <aicf/core/log.hpp>

#if AICF_ENABLE_NVTX
// NVTX를 실제로 쓰려면: <nvtx3/nvToolsExt.h> 포함 + 링크 설정 필요
// 지금은 “옵션 자리만” 마련. (추후 제대로 붙이자)
#endif

extern "C" void aicf_nvtx_anchor() {
#if AICF_ENABLE_NVTX
    aicf::log_info("NVTX enabled (stub)");
#else
    aicf::log_info("NVTX disabled");
#endif
}
