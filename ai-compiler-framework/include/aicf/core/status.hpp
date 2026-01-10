#pragma once
#include <cstdint>

namespace aicf {

enum class Status : int32_t {
  Ok = 0,
  Error = 1,
  NotImplemented = 2,
  InvalidArgument = 3,
  RuntimeError = 4,
  kInvalidArgument = 5,
  kCudaError = 6,
  kOk = 7,
};

constexpr bool ok(Status s) { return s == Status::Ok; }

constexpr const char* status_to_string(Status s) {
  switch (s) {
    case Status::Ok: return "Ok";
    case Status::Error: return "Error";
    case Status::NotImplemented: return "NotImplemented";
    case Status::InvalidArgument: return "InvalidArgument";
    default: return "Unknown";
  }
}

} // namespace aicf
