#pragma once
#include <torch/extension.h>
#include <aicf/core/status.hpp>
#include <aicf/runtime/stream.hpp>

inline bool is_cuda_f32_contig(const torch::Tensor& t) {
    return t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous();
}
inline bool status_ok(aicf::Status s) { return s == aicf::Status::Ok; }
inline aicf::Stream default_stream() { aicf::Stream s{}; s.handle = nullptr; return s; }
