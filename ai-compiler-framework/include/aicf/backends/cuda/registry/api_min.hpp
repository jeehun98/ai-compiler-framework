// include/aicf/backends/cuda/registry/api_min.hpp
#pragma once
#include <cstdint>
#include <string_view>

namespace aicf::cuda {

// ---- Status ----
enum class StatusCode : int32_t {
  kOk = 0,
  kInvalidArgument = 1,
  kNotSupported = 2,
  kInternal = 3,
  kCudaError = 4,
};

struct Status {
  StatusCode code;
  const char* msg;           // v0.1: static string or thread_local buffer
  constexpr bool ok() const { return code == StatusCode::kOk; }
  static constexpr Status Ok() { return {StatusCode::kOk, ""}; }
};

// ---- DType ----
enum class DType : int32_t { kF16 = 0, kF32 = 1 };

// ---- TensorDesc ----
// v0.1: contiguous only. stride는 일단 받아두되 검사에서 contiguous만 통과.
constexpr int kMaxRank = 8;

struct TensorDesc {
  void* data;                // device ptr
  DType dtype;
  int32_t rank;
  int64_t shape[kMaxRank];
  int64_t stride[kMaxRank];  // v0.1: contiguous stride만 허용
};

// ---- Stream ----
using CudaStream = void*;    // == cudaStream_t, 헤더 노출 최소화 목적

// ---- OpKind ----
enum class OpKind : int32_t {
  kAdd = 0,
  kRelu = 1,
  kGemm = 2,
  // ... 앞으로 여기만 늘어난다
};

// ---- Attr ----
// v0.1: attr는 (key -> int64/float/bool)만 지원. 문자열/배열은 v0.2.
enum class AttrTag : int32_t { kI64 = 0, kF32 = 1, kBool = 2 };

struct AttrValue {
  AttrTag tag;
  union {
    int64_t i64;
    float   f32;
    int32_t b32; // bool
  };
};

struct AttrKV {
  const char* key;
  AttrValue   val;
};

struct AttrMapView {
  const AttrKV* items;
  int32_t       size;

  // 단순 선형 탐색(v0.1). v0.2에서 정렬/해시로.
  bool get_i64(std::string_view k, int64_t* out) const;
  bool get_f32(std::string_view k, float* out) const;
  bool get_bool(std::string_view k, bool* out) const;
};

// ---- Unified op call ----
struct OpCall {
  OpKind kind;
  const TensorDesc* inputs;
  int32_t num_inputs;
  TensorDesc* outputs;
  int32_t num_outputs;
  AttrMapView attrs;
  CudaStream stream;         // nullptr이면 current stream
};

// Registry dispatch entry (C++)
Status Dispatch(const OpCall& call);

} // namespace aicf::cuda
