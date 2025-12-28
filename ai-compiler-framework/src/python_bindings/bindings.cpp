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
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>

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
// Graph state + stream policy
// ============================================================

struct CudaGraphState {
  bool capturing = false;
  bool captured  = false;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;

  // AICF owns this stream (created once). We will capture/launch on it.
  cudaStream_t aicf_stream = nullptr;

  // stream used for capture (== aicf_stream when capturing)
  cudaStream_t cap_stream = nullptr;

  void ensure_stream() {
    if (!aicf_stream) {
      // Non-blocking stream recommended
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
    // If we were capturing, try to end capture cleanly on cap_stream to restore state.
    if (cap_stream) {
      cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
      cudaError_t e = cudaStreamIsCapturing(cap_stream, &st);
      if (e == cudaSuccess && st != cudaStreamCaptureStatusNone) {
        cudaGraph_t tmp = nullptr;
        cudaError_t e2 = cudaStreamEndCapture(cap_stream, &tmp);
        if (e2 == cudaSuccess && tmp) cudaGraphDestroy(tmp);
        else cudaGetLastError();
      } else {
        cudaGetLastError();
      }
    }

    destroy_graph_objects();

    capturing = false;
    cap_stream = nullptr;
    // NOTE: keep aicf_stream alive across runs (cheap + stable).
  }

  ~CudaGraphState() {
    // best-effort cleanup at unload
    reset_full();
    if (aicf_stream) {
      cudaStreamDestroy(aicf_stream);
      aicf_stream = nullptr;
    }
  }
};

static CudaGraphState g_graph;
static std::mutex g_graph_mu;

// Stream policy used by op_call + replay:
// - if capturing/captured graph ops should run on aicf_stream (cap_stream)
// - otherwise use PyTorch current stream
static inline cudaStream_t aicf_dispatch_stream_locked() {
  if (g_graph.capturing && g_graph.cap_stream) return g_graph.cap_stream;
  // Replay can run on either stream, but running on aicf_stream is consistent.
  if (g_graph.captured && g_graph.aicf_stream) return g_graph.aicf_stream;
  return aicf_py::current_cuda_stream(); // PyTorch current stream
}

// ============================================================
// AttrPack builder (same as yours)
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
       + " status=" + aicf::status_to_string(st);
}

static void op_call_impl(
    aicf::cuda::OpKind kind,
    const py::sequence& inputs,
    const py::sequence& outputs,
    py::dict attrs) {

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

  cudaStream_t stream;
  {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    // ★ capture 중에는 AICF 전용 stream으로 enqueue
    stream = aicf_dispatch_stream_locked();
  }

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

PYBIND11_MODULE(_C, m) {
  m.doc() = "AICF CUDA unified bindings (Plan A): op_call";

  ensure_kernels_registered_once();

  py::enum_<aicf::cuda::OpKind>(m, "OpKind")
      .value("EltwiseAdd",  aicf::cuda::OpKind::EltwiseAdd)
      .value("EltwiseRelu", aicf::cuda::OpKind::EltwiseRelu)
      .value("Gemm",        aicf::cuda::OpKind::Gemm)
      .value("BiasAdd",     aicf::cuda::OpKind::BiasAdd)
      .value("ReduceSum",   aicf::cuda::OpKind::ReduceSum)
      .value("MseGrad",     aicf::cuda::OpKind::MseGrad)
      .value("ReluBwd",     aicf::cuda::OpKind::ReluBwd)
      .value("SgdStep",     aicf::cuda::OpKind::SgdStep)
      .value("Copy",        aicf::cuda::OpKind::Copy) // ✅ 추가

      .export_values();

  m.def(
    "op_call",
    [](aicf::cuda::OpKind kind,
       const py::sequence& inputs,
       const py::sequence& outputs,
       py::dict attrs) {
      op_call_impl(kind, inputs, outputs, attrs);
    },
    py::arg("kind"),
    py::arg("inputs"),
    py::arg("outputs"),
    py::arg("attrs") = py::dict()
  );

  m.def(
    "op_call",
    [](int kind,
       const py::sequence& inputs,
       const py::sequence& outputs,
       py::dict attrs) {
      op_call_impl(static_cast<aicf::cuda::OpKind>(kind), inputs, outputs, attrs);
    },
    py::arg("kind"),
    py::arg("inputs"),
    py::arg("outputs"),
    py::arg("attrs") = py::dict()
  );

  // ---------------- CUDA Graph control ----------------

  m.def("capture_begin", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);

    std::fprintf(stderr, "[aicf] capture_begin entered (dedicated stream)\n");

    g_graph.reset_full();
    g_graph.ensure_stream();

    // Capture on AICF-owned stream (NOT PyTorch current stream)
    const cudaStream_t s = g_graph.aicf_stream;

    // sanity: if somehow capturing, end it
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

    g_graph.cap_stream = s;
    g_graph.capturing = true;
  });

  m.def("capture_end", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);

    if (!g_graph.capturing || !g_graph.cap_stream) {
      throw std::runtime_error("capture_end called but not capturing");
    }

    cudaGraph_t graph = nullptr;
    cuda_check(cudaStreamEndCapture(g_graph.cap_stream, &graph),
               "cudaStreamEndCapture");

    g_graph.graph = graph;

    cudaGraphExec_t exec = nullptr;
    cuda_check(cudaGraphInstantiate(&exec, g_graph.graph, nullptr, nullptr, 0),
               "cudaGraphInstantiate");

    g_graph.exec = exec;
    g_graph.capturing = false;
    g_graph.captured  = true;
  });

  m.def("replay", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);

    if (!g_graph.captured || !g_graph.exec) {
      throw std::runtime_error("replay called but no captured graph exists");
    }

    // launch on same AICF stream for determinism
    g_graph.ensure_stream();
    cuda_check(cudaGraphLaunch(g_graph.exec, g_graph.aicf_stream), "cudaGraphLaunch");
  });

  m.def("capture_reset", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    g_graph.reset_full();
  });
}
