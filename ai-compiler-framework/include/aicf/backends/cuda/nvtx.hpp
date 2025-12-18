#pragma once

#ifndef AICF_ENABLE_NVTX
#define AICF_ENABLE_NVTX 0
#endif

#if AICF_ENABLE_NVTX
  // CUDA toolkit에 따라 둘 중 하나가 맞음
  #if __has_include(<nvtx3/nvToolsExt.h>)
    #include <nvtx3/nvToolsExt.h>
  #else
    #include <nvToolsExt.h>
  #endif
#endif

namespace aicf::cuda {

#if AICF_ENABLE_NVTX
struct NvtxRange {
  explicit NvtxRange(const char* msg) { nvtxRangePushA(msg); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};
#else
struct NvtxRange {
  explicit NvtxRange(const char*) {}
};
#endif

} // namespace aicf::cuda

#if AICF_ENABLE_NVTX
  #define AICF_NVTX_RANGE(MSG) ::aicf::cuda::NvtxRange _aicf_nvtx_range__(MSG)
#else
  #define AICF_NVTX_RANGE(MSG) ((void)0)
#endif
