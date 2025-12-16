#pragma once

#include <cstdint>

namespace aicf {

// 추후: cudaStream_t 래핑 + 캡처 safe 정책 포함
struct Stream {
    void* handle = nullptr; // cudaStream_t를 void*로 숨김 (헤더 의존 최소화)
};

} // namespace aicf
