#pragma once
#include <cstdint>

namespace aicf::cuda {

enum class Status : int32_t {
  Ok = 0,
  InvalidArgument = 1,
  NotImplemented = 2,
  Internal = 3,
};

inline bool ok(Status s) { return s == Status::Ok; }

inline const char* status_to_string(Status s) {
  switch (s) {
    case Status::Ok: return "Ok";
    case Status::InvalidArgument: return "InvalidArgument";
    case Status::NotImplemented: return "NotImplemented";
    case Status::Internal: return "Internal";
    default: return "Unknown";
  }
}

} // namespace aicf::cuda
