#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getDefaultCUDAStream 등


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

namespace py = pybind11;

// C ABI registry init (guarded)
static void ensure_kernels_registered_once() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    aicf_cuda_register_all_kernels();
  });
}

// AttrPack builder (v0.2)
// - supports bool / int / float only
// - SORT BY KEY to ensure deterministic packing
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
    aicf_py::check_tensor_v0_2(t, what);
    out.emplace_back(aicf_py::to_contig_desc_v0_2(t));
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

  // IMPORTANT: pass PyTorch current stream
  const cudaStream_t stream = aicf_py::current_cuda_stream();

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

static inline void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("[CUDA] ") + what + ": " + cudaGetErrorString(e));
  }
}

struct CudaGraphState {
  bool capturing = false;
  bool captured  = false;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;

  void reset() {
    if (exec)  { cudaGraphExecDestroy(exec); exec = nullptr; }
    if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
    capturing = false;
    captured  = false;
  }
};

static CudaGraphState g_graph;
static std::mutex g_graph_mu;


PYBIND11_MODULE(_C, m) {
  m.doc() = "AICF CUDA unified bindings (Plan A): op_call";

  ensure_kernels_registered_once();

  py::enum_<aicf::cuda::OpKind>(m, "OpKind")
      .value("EltwiseAdd",  aicf::cuda::OpKind::EltwiseAdd)
      .value("EltwiseRelu", aicf::cuda::OpKind::EltwiseRelu)
      .value("Gemm",        aicf::cuda::OpKind::Gemm)
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

    m.def("capture_begin", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);

    if (g_graph.capturing) {
      throw std::runtime_error("capture_begin called while already capturing");
    }

    // 기존 그래프가 있으면 날림 (원하면 유지 정책으로 바꿔도 됨)
    g_graph.reset();

    const cudaStream_t stream = aicf_py::current_cuda_stream();
    // Global 모드가 제일 단순. 필요하면 Relaxed/ThreadLocal로 바꿔도 됨.
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
               "cudaStreamBeginCapture");
    g_graph.capturing = true;
  });

  m.def("capture_end", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);

    if (!g_graph.capturing) {
      throw std::runtime_error("capture_end called but not capturing");
    }

    const cudaStream_t stream = aicf_py::current_cuda_stream();
    cudaGraph_t graph = nullptr;
    cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    g_graph.graph = graph;

    cudaGraphExec_t exec = nullptr;
    // flags는 0으로 시작. 필요하면 cudaGraphInstantiateFlagAutoFreeOnLaunch 등 검토.
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

    const cudaStream_t stream = aicf_py::current_cuda_stream();
    cuda_check(cudaGraphLaunch(g_graph.exec, stream), "cudaGraphLaunch");
  });

  m.def("capture_reset", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    g_graph.reset();
  });

}
