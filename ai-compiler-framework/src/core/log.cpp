#include <aicf/core/log.hpp>
#include <cstdio>
#include <cstdarg>

namespace aicf {

static void vlog(const char* level, const char* fmt, va_list ap) {
    std::fprintf(stderr, "[%s] ", level);
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
}

void log_info(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vlog("INFO", fmt, ap);
    va_end(ap);
}

void log_warn(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vlog("WARN", fmt, ap);
    va_end(ap);
}

void log_error(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vlog("ERROR", fmt, ap);
    va_end(ap);
}

} // namespace aicf
