// bindings.cpp (Unified CUDA bindings; primitive op_call + launch_by_id + CUDA Graph capture; AttrBlob ABI)
//
// core 없이 동작:
//  - Status는 aicf::cuda::Status (registry/status.hpp)
//  - attrs는 AttrBlob(schema_id + bytes + data)
//  - graph capture는 bindings 내부에서만 관리
//
// NEW:
//  - launch_by_id(kernel_id, kind, ...): decision-applied execution path (no runtime selection)
//    (OpKind prefix 추론 제거: bias_add 같은 multi-token prefix가 깨지기 때문)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <aicf/backends/cuda/registry/op_kind.hpp>
#include <aicf/backends/cuda/registry/dispatch.hpp>
#include <aicf/backends/cuda/registry/register_all.hpp>
#include <aicf/backends/cuda/registry/attr_blob.hpp>
#include <aicf/backends/cuda/registry/tensor_desc.hpp>
#include <aicf/backends/cuda/registry/status.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

// ------------------- registry init -------------------
static void ensure_kernels_registered_once() {
  static std::once_flag flag;
  std::call_once(flag, []() { aicf_cuda_register_all_kernels(); });
}

static inline void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("[CUDA] ") + what + ": " + cudaGetErrorString(e));
  }
}

// ============================================================
// Tensor helpers (common.hpp 대체)
// ============================================================

namespace aicf_py {

inline const char* dtype_name(const torch::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kHalf:  return "float16";
    case at::kFloat: return "float32";
    case at::kInt:   return "int32";
    default:         return "other";
  }
}

inline std::string tensor_brief(const torch::Tensor& t) {
  std::ostringstream oss;
  oss << "defined=" << (t.defined() ? "true" : "false");
  if (!t.defined()) return oss.str();

  oss << " device=" << (t.is_cuda() ? "cuda" : "cpu");
  oss << " dtype=" << dtype_name(t);
  oss << " contig=" << (t.is_contiguous() ? "true" : "false");
  oss << " rank=" << t.dim();
  oss << " shape=[";
  for (int i = 0; i < t.dim(); ++i) {
    oss << t.size(i);
    if (i + 1 < t.dim()) oss << ",";
  }
  oss << "]";
  return oss.str();
}

inline aicf::cuda::DType to_aicf_dtype_strict(const torch::Tensor& t) {
  if (t.scalar_type() == at::kHalf)  return aicf::cuda::DType::kF16;
  if (t.scalar_type() == at::kFloat) return aicf::cuda::DType::kF32;
  if (t.scalar_type() == at::kInt)   return aicf::cuda::DType::kI32;
  TORCH_CHECK(false, "unsupported dtype. got: ", tensor_brief(t));
}

inline void check_tensor_v0_3(const torch::Tensor& t, const char* what) {
  TORCH_CHECK(t.defined(), what, ": undefined tensor");
  TORCH_CHECK(t.is_cuda(), what, ": must be CUDA tensor. got: ", tensor_brief(t));

  const auto st = t.scalar_type();
  TORCH_CHECK(st == at::kHalf || st == at::kFloat || st == at::kInt,
              what, ": dtype must be float16/float32/int32. got: ", tensor_brief(t));

  const int64_t rank64 = t.dim();
  TORCH_CHECK(rank64 >= 0 && rank64 <= aicf::cuda::kMaxRank,
              what, ": rank out of range. got rank=", rank64,
              " (kMaxRank=", aicf::cuda::kMaxRank, ")");
}

inline aicf::cuda::TensorDesc to_desc_v0_3(const torch::Tensor& t) {
  check_tensor_v0_3(t, "to_desc_v0_3(t)");

  aicf::cuda::TensorDesc d{};
  d.data  = const_cast<void*>(t.data_ptr());
  d.dtype = to_aicf_dtype_strict(t);

  const int32_t r = static_cast<int32_t>(t.dim());
  d.r.rank = r;

  for (int i = 0; i < r; ++i) {
    d.shape[i]  = t.size(i);
    d.stride[i] = t.stride(i);
  }

  d.contiguous = t.is_contiguous();
  d.alignment  = 0;
  d.device     = 0;
  return d;
}

inline cudaStream_t current_cuda_stream() {
  return c10::cuda::getCurrentCUDAStream().stream();
}

} // namespace aicf_py

// ============================================================
// CUDA Graph state (singleton, minimal)
// ============================================================

struct CudaGraphState {
  bool capturing = false;
  bool captured  = false;

  cudaGraph_t     graph = nullptr;
  cudaGraphExec_t exec  = nullptr;

  cudaStream_t aicf_stream = nullptr; // dedicated stream used for capture + launch

  void ensure_stream() {
    if (!aicf_stream) {
      cuda_check(cudaStreamCreateWithFlags(&aicf_stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags");
    }
  }

  void destroy_graph_objects() {
    if (exec)  { cudaGraphExecDestroy(exec); exec = nullptr; }
    if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
    captured = false;
  }

  void reset_full() {
    if (aicf_stream) {
      cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
      cudaError_t e = cudaStreamIsCapturing(aicf_stream, &st);
      if (e == cudaSuccess && st != cudaStreamCaptureStatusNone) {
        cudaGraph_t tmp = nullptr;
        cudaError_t e2 = cudaStreamEndCapture(aicf_stream, &tmp);
        if (e2 == cudaSuccess && tmp) cudaGraphDestroy(tmp);
        else cudaGetLastError();
      } else {
        cudaGetLastError();
      }
    }

    destroy_graph_objects();
    capturing = false;
  }

  ~CudaGraphState() {
    reset_full();
    if (aicf_stream) {
      cudaStreamDestroy(aicf_stream);
      aicf_stream = nullptr;
    }
  }
};

static CudaGraphState g_graph;
static std::mutex g_graph_mu;

// ============================================================
// C++ op trace (optional)
// ============================================================

static std::vector<std::string> g_trace_ops;
static bool g_trace_enabled = true;

static inline const char* opkind_to_name(aicf::cuda::OpKind k) {
  switch (k) {
    case aicf::cuda::OpKind::EltwiseAdd:   return "add";
    case aicf::cuda::OpKind::EltwiseRelu:  return "relu";
    case aicf::cuda::OpKind::Gemm:         return "gemm";
    case aicf::cuda::OpKind::BiasAdd:      return "bias_add";
    case aicf::cuda::OpKind::ReduceSum:    return "reduce_sum";
    case aicf::cuda::OpKind::MseGrad:      return "mse_grad";
    case aicf::cuda::OpKind::ReluBwd:      return "relu_bwd";
    case aicf::cuda::OpKind::SgdStep:      return "sgd_step";
    case aicf::cuda::OpKind::Copy:         return "copy";
    case aicf::cuda::OpKind::GradZero:     return "grad_zero";
    case aicf::cuda::OpKind::AdamStep:     return "adam_step";
    case aicf::cuda::OpKind::StepInc:      return "step_inc";
    case aicf::cuda::OpKind::BiasCorr:     return "bias_corr";
    case aicf::cuda::OpKind::LayerNormFwd: return "layernorm_fwd";
    case aicf::cuda::OpKind::LayerNormBwd: return "layernorm_bwd";
    case aicf::cuda::OpKind::BatchNormFwd: return "batchnorm_fwd";
    case aicf::cuda::OpKind::BatchNormBwd: return "batchnorm_bwd";
    default: return "unknown";
  }
}

static inline void trace_record_locked(aicf::cuda::OpKind kind) {
  if (!g_trace_enabled) return;
  g_trace_ops.emplace_back(opkind_to_name(kind));
}

static inline void trace_record_kernel_id_locked(const std::string& kernel_id, aicf::cuda::OpKind kind) {
  if (!g_trace_enabled) return;
  g_trace_ops.emplace_back(std::string("kid:") + kernel_id + " kind:" + opkind_to_name(kind));
}

static inline void trace_reset_locked() { g_trace_ops.clear(); }

// ============================================================
// TensorDesc builder
// ============================================================

static void build_descs_v0_3(
    const py::sequence& seq,
    std::vector<aicf::cuda::TensorDesc>& out,
    const char* what) {

  out.clear();
  out.reserve((size_t)seq.size());

  for (auto h : seq) {
    torch::Tensor t = py::cast<torch::Tensor>(h);
    aicf_py::check_tensor_v0_3(t, what);
    out.emplace_back(aicf_py::to_desc_v0_3(t));
  }
}

static std::string op_fail_msg(aicf::cuda::OpKind kind, aicf::cuda::Status st) {
  return std::string("aicf_cuda call failed: kind=")
       + std::to_string((int)kind)
       + " (" + std::string(opkind_to_name(kind)) + ")"
       + " status=" + std::string(aicf::cuda::status_to_string(st))
       + " (" + std::to_string((int)st) + ")";
}

// ============================================================
// op_call primitive (attrs: schema_id + bytes, no dict)
// ============================================================

static void op_call_impl(
    aicf::cuda::OpKind kind,
    const py::sequence& inputs,
    const py::sequence& outputs,
    uint32_t schema_id,
    py::bytes attrs_bytes,
    uint64_t stream_u64 /* 0 => current stream */) {

  std::vector<aicf::cuda::TensorDesc> in_descs;
  std::vector<aicf::cuda::TensorDesc> out_descs;
  build_descs_v0_3(inputs,  in_descs,  "inputs");
  build_descs_v0_3(outputs, out_descs, "outputs");

  std::string buf = attrs_bytes;  // py::bytes -> std::string copy
  aicf::cuda::AttrBlob blob{};
  blob.schema_id = schema_id;
  blob.bytes = (uint32_t)buf.size();
  blob.data  = (buf.empty() ? nullptr : (const void*)buf.data());

  {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    trace_record_locked(kind);

    if (std::getenv("AICF_TRACE_STDERR")) {
      std::fprintf(stderr, "[aicf][op_call] %s (kind=%d) schema=0x%08x bytes=%u stream=0x%llx\n",
                   opkind_to_name(kind), (int)kind,
                   (unsigned)schema_id, (unsigned)blob.bytes,
                   (unsigned long long)stream_u64);
    }
  }

  cudaStream_t stream = stream_u64
      ? reinterpret_cast<cudaStream_t>(stream_u64)
      : aicf_py::current_cuda_stream();

  const aicf::cuda::Status st = aicf::cuda::dispatch_v0(
      kind,
      in_descs.data(),  (int32_t)in_descs.size(),
      out_descs.data(), (int32_t)out_descs.size(),
      (blob.bytes == 0 && blob.data == nullptr && blob.schema_id == 0) ? (const void*)nullptr
                                                                      : (const void*)&blob,
      stream);

  if (!aicf::cuda::ok(st)) {
    throw std::runtime_error(op_fail_msg(kind, st));
  }
}

// ============================================================
// launch_by_id (decision-applied path)  ✅ kind을 Python에서 받는다
// ============================================================

static void launch_by_id_impl(
    const std::string& kernel_id,
    aicf::cuda::OpKind kind,
    const py::sequence& inputs,
    const py::sequence& outputs,
    uint32_t schema_id,
    py::bytes attrs_bytes,
    uint64_t stream_u64 /* 0 => current stream */) {

  std::vector<aicf::cuda::TensorDesc> in_descs;
  std::vector<aicf::cuda::TensorDesc> out_descs;
  build_descs_v0_3(inputs,  in_descs,  "inputs");
  build_descs_v0_3(outputs, out_descs, "outputs");

  std::string buf = attrs_bytes;
  aicf::cuda::AttrBlob blob{};
  blob.schema_id = schema_id;
  blob.bytes = (uint32_t)buf.size();
  blob.data  = (buf.empty() ? nullptr : (const void*)buf.data());

  {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    trace_record_kernel_id_locked(kernel_id, kind);

    if (std::getenv("AICF_TRACE_STDERR")) {
      std::fprintf(stderr, "[aicf][launch_by_id] kernel_id=%s kind=%s schema=0x%08x bytes=%u stream=0x%llx\n",
                   kernel_id.c_str(),
                   opkind_to_name(kind),
                   (unsigned)schema_id,
                   (unsigned)blob.bytes,
                   (unsigned long long)stream_u64);
    }
  }

  cudaStream_t stream = stream_u64
      ? reinterpret_cast<cudaStream_t>(stream_u64)
      : aicf_py::current_cuda_stream();

  const aicf::cuda::Status st = aicf::cuda::dispatch_by_id_v0(
      kind,
      kernel_id.c_str(),
      in_descs.data(),  (int32_t)in_descs.size(),
      out_descs.data(), (int32_t)out_descs.size(),
      (blob.bytes == 0 && blob.data == nullptr && blob.schema_id == 0) ? (const void*)nullptr
                                                                      : (const void*)&blob,
      stream);

  if (!aicf::cuda::ok(st)) {
    throw std::runtime_error(op_fail_msg(kind, st) + " kernel_id=" + kernel_id);
  }
}

// ============================================================
// PYBIND module
// ============================================================

PYBIND11_MODULE(_C, m) {
  m.doc() = "AICF CUDA unified bindings (op_call + launch_by_id + CUDA Graph capture; AttrBlob ABI; core-free)";

  ensure_kernels_registered_once();

  py::enum_<aicf::cuda::OpKind>(m, "OpKind")
      .value("EltwiseAdd",   aicf::cuda::OpKind::EltwiseAdd)
      .value("EltwiseRelu",  aicf::cuda::OpKind::EltwiseRelu)
      .value("Gemm",         aicf::cuda::OpKind::Gemm)
      .value("BiasAdd",      aicf::cuda::OpKind::BiasAdd)
      .value("ReduceSum",    aicf::cuda::OpKind::ReduceSum)
      .value("MseGrad",      aicf::cuda::OpKind::MseGrad)
      .value("ReluBwd",      aicf::cuda::OpKind::ReluBwd)
      .value("SgdStep",      aicf::cuda::OpKind::SgdStep)
      .value("Copy",         aicf::cuda::OpKind::Copy)
      .value("GradZero",     aicf::cuda::OpKind::GradZero)
      .value("AdamStep",     aicf::cuda::OpKind::AdamStep)
      .value("StepInc",      aicf::cuda::OpKind::StepInc)
      .value("BiasCorr",     aicf::cuda::OpKind::BiasCorr)
      .value("LayerNormFwd", aicf::cuda::OpKind::LayerNormFwd)
      .value("LayerNormBwd", aicf::cuda::OpKind::LayerNormBwd)
      .value("BatchNormFwd", aicf::cuda::OpKind::BatchNormFwd)
      .value("BatchNormBwd", aicf::cuda::OpKind::BatchNormBwd)
      .export_values();

  // ---------------- op_call (legacy) ----------------
  m.def(
    "op_call",
    [](aicf::cuda::OpKind kind,
       const py::sequence& inputs,
       const py::sequence& outputs,
       uint32_t schema_id,
       py::bytes attrs_bytes,
       uint64_t stream) {
      op_call_impl(kind, inputs, outputs, schema_id, attrs_bytes, stream);
    },
    py::arg("kind"),
    py::arg("inputs"),
    py::arg("outputs"),
    py::arg("schema_id") = (uint32_t)0,
    py::arg("attrs_bytes") = py::bytes(),
    py::arg("stream") = (uint64_t)0
  );

  m.def(
    "op_call",
    [](int kind,
       const py::sequence& inputs,
       const py::sequence& outputs,
       uint32_t schema_id,
       py::bytes attrs_bytes,
       uint64_t stream) {
      op_call_impl(static_cast<aicf::cuda::OpKind>(kind), inputs, outputs, schema_id, attrs_bytes, stream);
    },
    py::arg("kind"),
    py::arg("inputs"),
    py::arg("outputs"),
    py::arg("schema_id") = (uint32_t)0,
    py::arg("attrs_bytes") = py::bytes(),
    py::arg("stream") = (uint64_t)0
  );

  // ---------------- launch_by_id (NEW) ----------------
  // ✅ OpKind을 prefix로 추론하지 않고 Python에서 받는다.
  m.def(
    "launch_by_id",
    [](const std::string& kernel_id,
       aicf::cuda::OpKind kind,
       const py::sequence& inputs,
       const py::sequence& outputs,
       uint32_t schema_id,
       py::bytes attrs_bytes,
       uint64_t stream) {
      launch_by_id_impl(kernel_id, kind, inputs, outputs, schema_id, attrs_bytes, stream);
    },
    py::arg("kernel_id"),
    py::arg("kind"),
    py::arg("inputs"),
    py::arg("outputs"),
    py::arg("schema_id") = (uint32_t)0,
    py::arg("attrs_bytes") = py::bytes(),
    py::arg("stream") = (uint64_t)0
  );

  // ---------------- trace API ----------------
  m.def("trace_enable", [](bool flag) {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    g_trace_enabled = flag;
  }, py::arg("flag") = true);

  m.def("trace_reset", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    trace_reset_locked();
  });

  m.def("trace_get", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    return g_trace_ops;
  });

  // ---------------- CUDA Graph control ----------------
  m.def("graph_begin", []() -> uint64_t {
    std::lock_guard<std::mutex> lock(g_graph_mu);

    std::fprintf(stderr, "[aicf] graph_begin (dedicated stream)\n");

    trace_reset_locked();

    g_graph.reset_full();
    g_graph.ensure_stream();

    const cudaStream_t s = g_graph.aicf_stream;

    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &st) == cudaSuccess && st != cudaStreamCaptureStatusNone) {
      cudaGraph_t tmp = nullptr;
      cudaStreamEndCapture(s, &tmp);
      if (tmp) cudaGraphDestroy(tmp);
      cudaGetLastError();
    } else {
      cudaGetLastError();
    }

    cuda_check(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal),
               "cudaStreamBeginCapture");
    g_graph.capturing = true;

    return (uint64_t)(reinterpret_cast<uintptr_t>(s));
  });

  m.def("graph_end", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);

    if (!g_graph.capturing) {
      throw std::runtime_error("graph_end called but not capturing");
    }

    cudaGraph_t graph = nullptr;
    cuda_check(cudaStreamEndCapture(g_graph.aicf_stream, &graph),
               "cudaStreamEndCapture");

    g_graph.graph = graph;

    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaGraphInstantiate(&exec, g_graph.graph, nullptr, nullptr, 0),
               "cudaGraphInstantiate");

    g_graph.exec = exec;
    g_graph.capturing = false;
    g_graph.captured  = true;
  });

  m.def("graph_launch", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);

    if (!g_graph.captured || !g_graph.exec) {
      throw std::runtime_error("graph_launch called but no captured graph exists");
    }

    g_graph.ensure_stream();
    cuda_check(cudaGraphLaunch(g_graph.exec, g_graph.aicf_stream), "cudaGraphLaunch");
  });

  m.def("graph_reset", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    g_graph.reset_full();
    trace_reset_locked();
  });

  m.def("graph_stream", []() -> uint64_t {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    if (!g_graph.aicf_stream) return 0;
    return (uint64_t)(reinterpret_cast<uintptr_t>(g_graph.aicf_stream));
  });
}
