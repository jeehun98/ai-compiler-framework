// bindings.cpp  (Unified CUDA bindings; primitive op_call + CUDA Graph capture)
//
// Goal of this version:
//  1) Keep bindings small and stable as ops grow:
//     - op_call is the only kernel-dispatch primitive
//     - NO IRExec / JSON parsing / lowering here (move to Python)
//  2) CUDA Graph capture is explicit and does NOT affect op_call implicitly:
//     - graph_begin() returns a dedicated cudaStream_t handle (as uint64)
//     - caller passes that stream into op_call(..., stream=...)
//  3) py::list has NO reserve(): never call reserve() on py::list
//
// Notes:
//  - AttrPack supports bool/int/float only (same as before)
//  - Graph API is singleton-based (minimal); you can later convert to handles if needed.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "common.hpp"

#include <aicf/backends/cuda/registry/op_kind.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>
#include <aicf/backends/cuda/registry/dispatch.hpp>
#include <aicf/backends/cuda/registry/register_all.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
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

  // best-effort cleanup if something left capturing
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

static inline void trace_reset_locked() { g_trace_ops.clear(); }

// ============================================================
// AttrPack builder (v0.2) + desc builder
// ============================================================

static void build_attr_pack_v0_2(
    const py::dict& attrs,
    std::vector<std::string>& key_storage,
    std::vector<aicf::cuda::AttrKV>& kv_storage,
    aicf::cuda::AttrPack& out_pack) {

  key_storage.clear();
  kv_storage.clear();

  if (attrs.size() == 0) {
    out_pack.items = nullptr;
    out_pack.size  = 0;
    return;
  }

  std::vector<std::pair<std::string, py::object>> items;
  items.reserve((size_t)attrs.size());

  for (auto it : attrs) {
    std::string k = py::cast<std::string>(it.first);
    py::object  v = py::reinterpret_borrow<py::object>(it.second);
    items.emplace_back(std::move(k), std::move(v));
  }

  std::sort(items.begin(), items.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  key_storage.reserve(items.size());
  kv_storage.reserve(items.size());

  for (auto& kvp : items) {
    key_storage.emplace_back(std::move(kvp.first));

    aicf::cuda::AttrKV kv{};
    kv.key = key_storage.back().c_str();

    const py::object& v = kvp.second;

    if (py::isinstance<py::bool_>(v)) {
      kv.val.tag = aicf::cuda::AttrTag::kBool;
      kv.val.b32 = py::cast<bool>(v) ? 1 : 0;
    } else if (py::isinstance<py::int_>(v)) {
      kv.val.tag = aicf::cuda::AttrTag::kI64;
      kv.val.i64 = py::cast<int64_t>(v);
    } else if (py::isinstance<py::float_>(v)) {
      kv.val.tag = aicf::cuda::AttrTag::kF32;
      kv.val.f32 = py::cast<float>(v);
    } else {
      throw std::runtime_error(
        "aicf_cuda.op_call: attrs supports only bool/int/float (binding v0.2)");
    }

    kv_storage.emplace_back(kv);
  }

  out_pack.items = kv_storage.data();
  out_pack.size  = static_cast<int32_t>(kv_storage.size());
}

static void build_descs_v0_2(
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

static std::string op_fail_msg(aicf::cuda::OpKind kind, aicf::Status st) {
  return std::string("aicf_cuda.op_call failed: kind=")
       + std::to_string((int)kind)
       + " (" + std::string(opkind_to_name(kind)) + ")"
       + " status=" + aicf::status_to_string(st);
}

// ============================================================
// op_call primitive (NO implicit stream policy)
// ============================================================

static void op_call_impl(
    aicf::cuda::OpKind kind,
    const py::sequence& inputs,
    const py::sequence& outputs,
    py::dict attrs,
    uint64_t stream_u64 /* 0 => current stream */) {

  std::vector<aicf::cuda::TensorDesc> in_descs;
  std::vector<aicf::cuda::TensorDesc> out_descs;
  build_descs_v0_2(inputs,  in_descs,  "inputs");
  build_descs_v0_2(outputs, out_descs, "outputs");

  aicf::cuda::AttrPack pack{};
  std::vector<std::string> key_storage;
  std::vector<aicf::cuda::AttrKV> kv_storage;
  const void* attr_ptr = nullptr;

  if (attrs.size() > 0) {
    build_attr_pack_v0_2(attrs, key_storage, kv_storage, pack);
    attr_ptr = &pack;
  }

  // record trace (no stream policy here)
  {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    trace_record_locked(kind);

    if (std::getenv("AICF_TRACE_STDERR")) {
      std::fprintf(stderr, "[aicf][op_call] %s (kind=%d) stream=0x%llx\n",
                   opkind_to_name(kind), (int)kind,
                   (unsigned long long)stream_u64);
    }
  }

  cudaStream_t stream = stream_u64
      ? reinterpret_cast<cudaStream_t>(stream_u64)
      : aicf_py::current_cuda_stream();

  const aicf::Status st = aicf::cuda::dispatch_v0(
      kind,
      in_descs.data(),  (int)in_descs.size(),
      out_descs.data(), (int)out_descs.size(),
      attr_ptr,
      stream);

  if (!aicf::ok(st)) {
    throw std::runtime_error(op_fail_msg(kind, st));
  }
}

// ============================================================
// PYBIND module
// ============================================================

PYBIND11_MODULE(_C, m) {
  m.doc() = "AICF CUDA unified bindings (primitive op_call + CUDA Graph capture)";

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

  // ---------------- op_call ----------------
  // op_call(kind_enum, inputs, outputs, attrs={}, stream=0)
  m.def(
    "op_call",
    [](aicf::cuda::OpKind kind,
       const py::sequence& inputs,
       const py::sequence& outputs,
       py::dict attrs,
       uint64_t stream) {
      op_call_impl(kind, inputs, outputs, attrs, stream);
    },
    py::arg("kind"),
    py::arg("inputs"),
    py::arg("outputs"),
    py::arg("attrs") = py::dict(),
    py::arg("stream") = (uint64_t)0
  );

  // op_call(kind_int, inputs, outputs, attrs={}, stream=0)
  m.def(
    "op_call",
    [](int kind,
       const py::sequence& inputs,
       const py::sequence& outputs,
       py::dict attrs,
       uint64_t stream) {
      op_call_impl(static_cast<aicf::cuda::OpKind>(kind), inputs, outputs, attrs, stream);
    },
    py::arg("kind"),
    py::arg("inputs"),
    py::arg("outputs"),
    py::arg("attrs") = py::dict(),
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
    return g_trace_ops; // copy
  });

  // ---------------- CUDA Graph control ----------------
  // graph_begin() -> uint64 stream_handle
  // Caller must pass returned stream_handle to op_call(..., stream=handle)
  m.def("graph_begin", []() -> uint64_t {
    std::lock_guard<std::mutex> lock(g_graph_mu);

    std::fprintf(stderr, "[aicf] graph_begin (dedicated stream)\n");

    trace_reset_locked();

    g_graph.reset_full();
    g_graph.ensure_stream();

    const cudaStream_t s = g_graph.aicf_stream;

    // defensive cleanup if stream is already capturing (shouldn't happen)
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

    // return stream pointer as uint64_t
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

  // Optional: expose the dedicated stream handle when captured (for debugging)
  m.def("graph_stream", []() -> uint64_t {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    if (!g_graph.aicf_stream) return 0;
    return (uint64_t)(reinterpret_cast<uintptr_t>(g_graph.aicf_stream));
  });
}
