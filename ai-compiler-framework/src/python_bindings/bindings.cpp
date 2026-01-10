// bindings.cpp  (IR-only executable prototype; JSON parsed via Python's json module)
//
// What this adds on top of your existing bindings:
//  - compile_ir_json(ir_json) -> handle
//  - exe_bind(handle, name, tensor)
//  - exe_required_inputs(handle) -> [names]
//  - exe_dump(handle) -> dict (ops + bindings + required inputs)
//  - exe_run_once(handle)  (eager run on current stream policy)
//  - exe_capture(handle)   (capture_begin + run_once + capture_end)
//  - exe_replay(handle)
//
// Notes:
//  - This prototype lowers only these IR ops:
//      Linear, ReLU, MseGrad, GradZero, StepInc, BiasCorr, AdamStep
//    Backward is NOT lowered here (yet). If your IR contains AdamStep grads
//    produced by Backward, you must bind those grad tensors explicitly (dW/db etc)
//    before running the optim slice.
//  - Intermediates (e.g., linear_out / relu_out / mse_grad_out) are lazily allocated
//    using IR value meta (shape/dtype/device). External inputs (x/t/W/b/m/v/grad/step/bc1/bc2)
//    must be bound by user.
//
// This is intentionally "minimal diff": it reuses your op_call_impl + CUDA graph controls.

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
#include <unordered_map>
#include <unordered_set>
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
// Graph state + stream policy (your original code, unchanged)
// ============================================================

struct CudaGraphState {
  bool capturing = false;
  bool captured  = false;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;

  cudaStream_t aicf_stream = nullptr;
  cudaStream_t cap_stream = nullptr;

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
// NEW: C++ op trace (authoritative) (your original code, unchanged)
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

// Stream policy used by op_call + replay:
static inline cudaStream_t aicf_dispatch_stream_locked() {
  if (g_graph.capturing && g_graph.cap_stream) return g_graph.cap_stream;
  if (g_graph.captured && g_graph.aicf_stream) return g_graph.aicf_stream;
  return aicf_py::current_cuda_stream();
}

// ============================================================
// AttrPack builder (v0.2) (your original code, unchanged)
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

    // record trace
    trace_record_locked(kind);

    // optional stderr trace
    if (std::getenv("AICF_TRACE_STDERR")) {
      std::fprintf(stderr, "[aicf][op_call] %s (kind=%d) capturing=%d captured=%d\n",
                   opkind_to_name(kind), (int)kind,
                   (int)g_graph.capturing, (int)g_graph.captured);
    }

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

// ============================================================
// IR-only Executable (NEW)
// ============================================================

struct ValMeta {
  std::vector<int64_t> shape;
  std::string dtype;   // e.g. "torch.float32"
  std::string device;  // e.g. "cuda:0"
};

struct LoweredOp {
  aicf::cuda::OpKind kind;
  std::vector<std::string> ins;
  std::vector<std::string> outs;
  py::dict attrs;
};

static inline std::string to_lower(std::string s) {
  for (char& c : s) c = (char)std::tolower((unsigned char)c);
  return s;
}

static inline int parse_cuda_index(const std::string& dev) {
  // "cuda:0" / "cuda" / "cuda:1"
  auto p = dev.find(':');
  if (p == std::string::npos) return 0;
  return std::max(0, std::atoi(dev.c_str() + (int)p + 1));
}

static inline torch::Dtype parse_torch_dtype(const std::string& dt) {
  std::string s = to_lower(dt);
  // accept "torch.float32", "float32", etc.
  if (s.find("float32") != std::string::npos || s.find("f32") != std::string::npos) return torch::kFloat32;
  if (s.find("float16") != std::string::npos || s.find("f16") != std::string::npos || s.find("half") != std::string::npos) return torch::kFloat16;
  if (s.find("bfloat16") != std::string::npos || s.find("bf16") != std::string::npos) return torch::kBFloat16;
  if (s.find("int32") != std::string::npos || s.find("i32") != std::string::npos) return torch::kInt32;
  if (s.find("int64") != std::string::npos || s.find("i64") != std::string::npos) return torch::kInt64;
  throw std::runtime_error("compile_ir_json: unsupported dtype string: " + dt);
}

static inline torch::Device parse_torch_device(const std::string& dev) {
  std::string s = to_lower(dev);
  if (s.find("cuda") != std::string::npos) {
    return torch::Device(torch::kCUDA, parse_cuda_index(s));
  }
  if (s.find("cpu") != std::string::npos) {
    return torch::Device(torch::kCPU);
  }
  throw std::runtime_error("compile_ir_json: unsupported device string: " + dev);
}

struct IRExec {
  std::string graph_name;
  std::vector<LoweredOp> ops;

  // name -> bound tensor (external or allocated)
  std::unordered_map<std::string, torch::Tensor> bind;

  // name -> meta (for lazy alloc)
  std::unordered_map<std::string, ValMeta> meta;

  // required external inputs (computed at compile time)
  std::vector<std::string> required_inputs;

  // allocate lazily for intermediates
  torch::Tensor get_or_alloc(const std::string& name) {
    auto it = bind.find(name);
    if (it != bind.end()) return it->second;

    auto mit = meta.find(name);
    if (mit == meta.end()) {
      throw std::runtime_error("IRExec: tensor not bound and no meta for name=" + name);
    }

    const ValMeta& m = mit->second;
    torch::Dtype dt = parse_torch_dtype(m.dtype);
    torch::Device dv = parse_torch_device(m.device);

    torch::TensorOptions opt = torch::TensorOptions().dtype(dt).device(dv);
    std::vector<int64_t> sz = m.shape;

    // scalars: allow empty shape -> []
    torch::Tensor t = torch::empty(sz, opt);
    bind.emplace(name, t);
    return t;
  }

  // eager run (uses op_call_impl -> dispatch_v0, obeys stream policy)
  void run_once() {
    for (const auto& op : ops) {
      py::list in_list;
      py::list out_list;

      in_list.attr("reserve")(op.ins.size());
      out_list.attr("reserve")(op.outs.size());

      for (const auto& n : op.ins) {
        in_list.append(get_or_alloc(n));
      }
      for (const auto& n : op.outs) {
        out_list.append(get_or_alloc(n));
      }

      op_call_impl(op.kind, in_list, out_list, op.attrs);
    }
  }
};

// exec registry
static std::mutex g_exec_mu;
static int g_next_exec_id = 1;
static std::unordered_map<int, std::shared_ptr<IRExec>> g_execs;

// ============================================================
// IR JSON parsing + lowering (NEW, dependency-free)
// ============================================================

static py::object py_json_loads() {
  static py::object loads;
  static std::once_flag flag;
  std::call_once(flag, []() {
    py::module_ json = py::module_::import("json");
    loads = json.attr("loads");
  });
  return loads;
}

static aicf::cuda::OpKind lowered_kind_from_name(const std::string& op) {
  const std::string s = to_lower(op);

  // lowered kernel names (runtime)
  if (s == "gemm") return aicf::cuda::OpKind::Gemm;
  if (s == "bias_add") return aicf::cuda::OpKind::BiasAdd;
  if (s == "relu") return aicf::cuda::OpKind::EltwiseRelu;
  if (s == "mse_grad") return aicf::cuda::OpKind::MseGrad;
  if (s == "grad_zero") return aicf::cuda::OpKind::GradZero;
  if (s == "step_inc") return aicf::cuda::OpKind::StepInc;
  if (s == "bias_corr") return aicf::cuda::OpKind::BiasCorr;
  if (s == "adam_step") return aicf::cuda::OpKind::AdamStep;

  throw std::runtime_error("IR lowering: unknown lowered op name: " + op);
}

// Lower high-level IR op -> lowered ops
static void lower_ir_node_into(
    const std::string& ir_op,
    const std::vector<std::string>& in_names,
    const std::vector<std::string>& out_names,
    const py::dict& attrs,
    std::vector<LoweredOp>& out_ops) {

  const std::string op = to_lower(ir_op);

  if (op == "linear") {
    // IR: inputs [x, W, (b)] outputs [y]
    // Lower: gemm(x, W) -> y   (transB=True)
    //        optional bias_add(y, b) -> y (in-place)
    if (out_names.size() != 1) throw std::runtime_error("Linear: expects 1 output");
    if (!(in_names.size() == 2 || in_names.size() == 3)) throw std::runtime_error("Linear: expects 2 or 3 inputs");

    {
      LoweredOp g{};
      g.kind = aicf::cuda::OpKind::Gemm;
      g.ins = {in_names[0], in_names[1]};
      g.outs = {out_names[0]};
      g.attrs = py::dict();
      g.attrs["transB"] = true;
      out_ops.emplace_back(std::move(g));
    }

    bool has_bias = false;
    if (attrs.contains("bias")) {
      has_bias = py::cast<bool>(attrs["bias"]);
    } else if (in_names.size() == 3) {
      has_bias = true;
    }

    if (has_bias) {
      if (in_names.size() < 3) throw std::runtime_error("Linear: bias=true but no bias input");
      LoweredOp b{};
      b.kind = aicf::cuda::OpKind::BiasAdd;
      b.ins = {out_names[0], in_names[2]};
      b.outs = {out_names[0]}; // in-place
      b.attrs = py::dict();
      out_ops.emplace_back(std::move(b));
    }
    return;
  }

  if (op == "relu") {
    // IR: inputs [x] outputs [y]
    if (in_names.size() != 1 || out_names.size() != 1) throw std::runtime_error("ReLU: expects 1 in/1 out");
    LoweredOp r{};
    r.kind = aicf::cuda::OpKind::EltwiseRelu;
    r.ins = {in_names[0]};
    r.outs = {out_names[0]};
    r.attrs = py::dict();
    out_ops.emplace_back(std::move(r));
    return;
  }

  if (op == "msegrad" || op == "mse_grad") {
    // IR: inputs [pred, target] outputs [grad]
    if (in_names.size() != 2 || out_names.size() != 1) throw std::runtime_error("MseGrad: expects 2 in/1 out");
    LoweredOp m{};
    m.kind = aicf::cuda::OpKind::MseGrad;
    m.ins = {in_names[0], in_names[1]};
    m.outs = {out_names[0]};
    m.attrs = py::dict();
    if (attrs.contains("scale")) m.attrs["scale"] = attrs["scale"];
    out_ops.emplace_back(std::move(m));
    return;
  }

  if (op == "gradzero" || op == "grad_zero") {
    // IR: inputs [g] outputs [g] (in-place semantics)
    if (in_names.size() != 1 || out_names.size() != 1) throw std::runtime_error("GradZero: expects 1 in/1 out");
    LoweredOp z{};
    z.kind = aicf::cuda::OpKind::GradZero;
    z.ins = {in_names[0]};
    z.outs = {out_names[0]};
    z.attrs = py::dict();
    out_ops.emplace_back(std::move(z));
    return;
  }

  if (op == "stepinc" || op == "step_inc") {
    if (in_names.size() != 1 || out_names.size() != 1) throw std::runtime_error("StepInc: expects 1 in/1 out");
    LoweredOp s{};
    s.kind = aicf::cuda::OpKind::StepInc;
    s.ins = {in_names[0]};
    s.outs = {out_names[0]};
    s.attrs = py::dict();
    out_ops.emplace_back(std::move(s));
    return;
  }

  if (op == "biascorr" || op == "bias_corr") {
    if (in_names.size() != 1 || out_names.size() != 2) throw std::runtime_error("BiasCorr: expects 1 in/2 out");
    LoweredOp b{};
    b.kind = aicf::cuda::OpKind::BiasCorr;
    b.ins = {in_names[0]};
    b.outs = {out_names[0], out_names[1]};
    b.attrs = py::dict();
    if (attrs.contains("beta1")) b.attrs["beta1"] = attrs["beta1"];
    if (attrs.contains("beta2")) b.attrs["beta2"] = attrs["beta2"];
    out_ops.emplace_back(std::move(b));
    return;
  }

  if (op == "adamstep" || op == "adam_step") {
    // inputs: [p, grad, m, v, bc1_inv, bc2_inv] outputs: [p, m, v]
    if (in_names.size() != 6 || out_names.size() != 3) throw std::runtime_error("AdamStep: expects 6 in/3 out");
    LoweredOp a{};
    a.kind = aicf::cuda::OpKind::AdamStep;
    a.ins = in_names;
    a.outs = out_names;
    a.attrs = py::dict();
    if (attrs.contains("lr"))    a.attrs["lr"] = attrs["lr"];
    if (attrs.contains("beta1")) a.attrs["beta1"] = attrs["beta1"];
    if (attrs.contains("beta2")) a.attrs["beta2"] = attrs["beta2"];
    if (attrs.contains("eps"))   a.attrs["eps"] = attrs["eps"];
    out_ops.emplace_back(std::move(a));
    return;
  }

  if (op == "backward") {
    // Not supported in this minimal prototype.
    // You can still IR-run forward/optim slices if you bind grads explicitly.
    throw std::runtime_error("IR lowering: Backward is not supported in this bindings-only prototype yet.");
  }

  throw std::runtime_error("IR lowering: unsupported IR op: " + ir_op);
}

static std::shared_ptr<IRExec> compile_ir_json_impl(const std::string& ir_json) {
  py::object loads = py_json_loads();
  py::object root_obj = loads(py::str(ir_json));
  py::dict root = py::cast<py::dict>(root_obj);

  auto exe = std::make_shared<IRExec>();

  if (root.contains("graph")) exe->graph_name = py::cast<std::string>(root["graph"]);
  else exe->graph_name = "graph";

  // values: map(str(id) -> {id,name,shape,dtype,device})
  if (!root.contains("values")) throw std::runtime_error("compile_ir_json: missing 'values'");
  py::dict values = py::cast<py::dict>(root["values"]);

  // build id -> name and meta(name -> meta)
  int max_id = -1;
  for (auto it : values) {
    py::dict v = py::cast<py::dict>(it.second);
    int vid = py::cast<int>(v["id"]);
    max_id = std::max(max_id, vid);
  }

  std::vector<std::string> id2name((size_t)max_id + 1);
  std::vector<bool> id_valid((size_t)max_id + 1, false);

  for (auto it : values) {
    py::dict v = py::cast<py::dict>(it.second);
    int vid = py::cast<int>(v["id"]);
    std::string name = py::cast<std::string>(v["name"]);
    id2name[(size_t)vid] = name;
    id_valid[(size_t)vid] = true;

    ValMeta m;
    // shape is list
    py::list shp = py::cast<py::list>(v["shape"]);
    for (auto s : shp) m.shape.push_back(py::cast<int64_t>(s));
    m.dtype = py::cast<std::string>(v["dtype"]);
    m.device = py::cast<std::string>(v["device"]);
    exe->meta.emplace(name, std::move(m));
  }

  // nodes
  if (!root.contains("nodes")) throw std::runtime_error("compile_ir_json: missing 'nodes'");
  py::list nodes = py::cast<py::list>(root["nodes"]);

  // For required_inputs analysis:
  // - produced_names: any output of a lowered op
  // - used_names: any input of a lowered op
  std::unordered_set<std::string> produced;
  std::unordered_set<std::string> used;

  // build lowered ops
  for (auto nobj : nodes) {
    py::dict n = py::cast<py::dict>(nobj);
    std::string op = py::cast<std::string>(n["op"]);

    py::list ins = py::cast<py::list>(n["inputs"]);
    py::list outs = py::cast<py::list>(n["outputs"]);
    py::dict attrs = n.contains("attrs") ? py::cast<py::dict>(n["attrs"]) : py::dict();

    std::vector<std::string> in_names;
    std::vector<std::string> out_names;

    in_names.reserve((size_t)ins.size());
    out_names.reserve((size_t)outs.size());

    for (auto x : ins) {
      int vid = py::cast<int>(x);
      if (vid < 0 || (size_t)vid >= id2name.size() || !id_valid[(size_t)vid]) {
        throw std::runtime_error("compile_ir_json: invalid input vid=" + std::to_string(vid));
      }
      in_names.emplace_back(id2name[(size_t)vid]);
    }
    for (auto x : outs) {
      int vid = py::cast<int>(x);
      if (vid < 0 || (size_t)vid >= id2name.size() || !id_valid[(size_t)vid]) {
        throw std::runtime_error("compile_ir_json: invalid output vid=" + std::to_string(vid));
      }
      out_names.emplace_back(id2name[(size_t)vid]);
    }

    std::vector<LoweredOp> tmp;
    tmp.reserve(2);
    lower_ir_node_into(op, in_names, out_names, attrs, tmp);

    for (auto& lop : tmp) {
      for (auto& nm : lop.ins) used.insert(nm);
      for (auto& nm : lop.outs) produced.insert(nm);
      exe->ops.emplace_back(std::move(lop));
    }
  }

  // required inputs = used - produced
  // (These must be provided via exe_bind; intermediates will still be alloc'ed lazily,
  //  but if something is truly external and you didn't bind it, you'll silently get garbage.
  //  So we surface them here.)
  {
    std::vector<std::string> req;
    req.reserve(used.size());
    for (const auto& nm : used) {
      if (produced.find(nm) == produced.end()) req.push_back(nm);
    }
    std::sort(req.begin(), req.end());
    exe->required_inputs = std::move(req);
  }

  return exe;
}

// ============================================================
// PYBIND module
// ============================================================

PYBIND11_MODULE(_C, m) {
  m.doc() = "AICF CUDA unified bindings + IR-only executable prototype";

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
      .value("Copy",        aicf::cuda::OpKind::Copy)
      .value("GradZero",    aicf::cuda::OpKind::GradZero)
      .value("AdamStep",    aicf::cuda::OpKind::AdamStep)
      .value("StepInc",     aicf::cuda::OpKind::StepInc)
      .value("BiasCorr",    aicf::cuda::OpKind::BiasCorr)
      .value("LayerNormFwd", aicf::cuda::OpKind::LayerNormFwd)
      .value("LayerNormBwd", aicf::cuda::OpKind::LayerNormBwd)
      .value("BatchNormFwd", aicf::cuda::OpKind::BatchNormFwd)
      .value("BatchNormBwd", aicf::cuda::OpKind::BatchNormBwd)
      .export_values();

  // ---------------- op_call ----------------
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
  m.def("capture_begin", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);

    std::fprintf(stderr, "[aicf] capture_begin entered (dedicated stream)\n");

    // clear trace at capture begin
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

    g_graph.ensure_stream();
    cuda_check(cudaGraphLaunch(g_graph.exec, g_graph.aicf_stream), "cudaGraphLaunch");
  });

  m.def("capture_reset", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    g_graph.reset_full();
    trace_reset_locked();
  });

  // ============================================================
  // IR-only executable API (NEW)
  // ============================================================

  m.def("compile_ir_json", [](const std::string& ir_json) -> int {
    auto exe = compile_ir_json_impl(ir_json);
    std::lock_guard<std::mutex> lock(g_exec_mu);
    const int hid = g_next_exec_id++;
    g_execs.emplace(hid, std::move(exe));
    return hid;
  }, py::arg("ir_json"));

  m.def("exe_destroy", [](int handle) {
    std::lock_guard<std::mutex> lock(g_exec_mu);
    g_execs.erase(handle);
  }, py::arg("handle"));

  m.def("exe_bind", [](int handle, const std::string& name, const torch::Tensor& t) {
    aicf_py::check_tensor_v0_3(t, "exe_bind tensor");
    std::lock_guard<std::mutex> lock(g_exec_mu);
    auto it = g_execs.find(handle);
    if (it == g_execs.end()) throw std::runtime_error("exe_bind: invalid handle");
    it->second->bind[name] = t;
  }, py::arg("handle"), py::arg("name"), py::arg("tensor"));

  m.def("exe_required_inputs", [](int handle) {
    std::lock_guard<std::mutex> lock(g_exec_mu);
    auto it = g_execs.find(handle);
    if (it == g_execs.end()) throw std::runtime_error("exe_required_inputs: invalid handle");
    return it->second->required_inputs;
  }, py::arg("handle"));

  m.def("exe_dump", [](int handle) {
    std::lock_guard<std::mutex> lock(g_exec_mu);
    auto it = g_execs.find(handle);
    if (it == g_execs.end()) throw std::runtime_error("exe_dump: invalid handle");
    auto& exe = *it->second;

    py::dict d;
    d["graph"] = exe.graph_name;

    py::list ops;
    for (const auto& op : exe.ops) {
      py::dict od;
      od["kind"] = (int)op.kind;
      od["op"] = std::string(opkind_to_name(op.kind));
      od["inputs"] = op.ins;
      od["outputs"] = op.outs;
      od["attrs"] = op.attrs;
      ops.append(od);
    }
    d["ops"] = ops;

    py::list bound;
    bound.attr("reserve")(exe.bind.size());
    for (const auto& kv : exe.bind) bound.append(kv.first);
    d["bound"] = bound;

    d["required_inputs"] = exe.required_inputs;

    return d;
  }, py::arg("handle"));

  m.def("exe_run_once", [](int handle) {
    std::shared_ptr<IRExec> exe;
    {
      std::lock_guard<std::mutex> lock(g_exec_mu);
      auto it = g_execs.find(handle);
      if (it == g_execs.end()) throw std::runtime_error("exe_run_once: invalid handle");
      exe = it->second;
    }
    exe->run_once();
  }, py::arg("handle"));

  m.def("exe_capture", [](int handle) {
    std::shared_ptr<IRExec> exe;
    {
      std::lock_guard<std::mutex> lock(g_exec_mu);
      auto it = g_execs.find(handle);
      if (it == g_execs.end()) throw std::runtime_error("exe_capture: invalid handle");
      exe = it->second;
    }

    // reuse your capture lifecycle
    {
      std::lock_guard<std::mutex> lock(g_graph_mu);
      // clear trace at capture begin
      trace_reset_locked();
    }

    // begin capture
    {
      std::lock_guard<std::mutex> lock(g_graph_mu);
      std::fprintf(stderr, "[aicf] capture_begin entered (dedicated stream)\n");
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

      g_graph.cap_stream = s;
      g_graph.capturing = true;
    }

    // record ops into the capture stream
    exe->run_once();

    // end capture
    {
      std::lock_guard<std::mutex> lock(g_graph_mu);

      if (!g_graph.capturing || !g_graph.cap_stream) {
        throw std::runtime_error("exe_capture: internal error (not capturing)");
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
    }
  }, py::arg("handle"));

  m.def("exe_replay", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);

    if (!g_graph.captured || !g_graph.exec) {
      throw std::runtime_error("exe_replay called but no captured graph exists");
    }

    g_graph.ensure_stream();
    cuda_check(cudaGraphLaunch(g_graph.exec, g_graph.aicf_stream), "cudaGraphLaunch");
  });

  m.def("exe_capture_reset", []() {
    std::lock_guard<std::mutex> lock(g_graph_mu);
    g_graph.reset_full();
    trace_reset_locked();
  });
}
