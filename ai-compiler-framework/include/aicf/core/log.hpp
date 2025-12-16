#pragma once

#include <cstdarg>

namespace aicf {

// 최소 로깅 API (나중에 spdlog로 바꾸든, NVTX로 보내든 확장)
void log_info(const char* fmt, ...);
void log_warn(const char* fmt, ...);
void log_error(const char* fmt, ...);

} // namespace aicf
