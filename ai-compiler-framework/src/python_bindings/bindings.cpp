// src/python_bindings/bindings.cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "common.hpp"

#include <aicf/backends/cuda/registry/op_kind.hpp>
#include <aicf/backends/cuda/registry/attr_pack.hpp>
#include <aicf/backends/cuda/registry/dispatch.hpp>
#include <aicf/backends/cuda/registry/register_all.hpp>

#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <mutex>

namespace py = pybind11;

// C ABI registry init (guarded)
static void ensure_kernels_registered_once() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    aicf_cuda_register_all_kernels();
  });
}

// -------------------------
// AttrPack builder (v0.1)
// -------------------------
static void build_attr_pack_v0(
    const py::dict& attrs,
    std::vector<std::string>& key_storage,
    std::vector<aicf::cuda::AttrKV>& kv_storage,
    aicf::cuda::AttrPack& out_pack) {

  key_storage.clear();
  kv_storage.clear();

  key_storage.reserve((size_t)attrs.size());
  kv_storage.reserve((size_t)attrs.size());

  for (auto item : attrs) {
    std::string k = py::cast<std::string>(item.first);
    key_storage.emplace_back(std::move(k));

    aicf::cuda::AttrKV kv{};
    kv.key = key_storage.back().c_str();

    const py::handle v = item.second;

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
        "aicf_cuda.op_call: attrs supports only bool/int/float (v0.1)");
    }

    kv_storage.emplace_back(kv);
  }

  out_pack.items = kv_storage.empty() ? nullptr : kv_storage.data();
  out_pack.size  = static_cast<int32_t>(kv_storage.size());
}

// -------------------------
// TensorDesc builder
// -------------------------
static void build_descs_v0(
    const py::sequence& seq,
    std::vector<aicf::cuda::TensorDesc>& out,
    const char* what) {

  out.clear();
  out.reserve((size_t)seq.size());

  for (auto h : seq) {
    torch::Tensor t = py::cast<torch::Tensor>(h);
    aicf_py::check_tensor_v0(t, what);
    out.emplace_back(aicf_py::to_desc_v0(t));
  }
}

// shared implementation for both overloads
static void op_call_impl(
    aicf::cuda::OpKind kind,
    const py::sequence& inputs,
    const py::sequence& outputs,
    py::dict attrs) {

  std::vector<aicf::cuda::TensorDesc> in_descs;
  std::vector<aicf::cuda::TensorDesc> out_descs;
  build_descs_v0(inputs,  in_descs,  "inputs");
  build_descs_v0(outputs, out_descs, "outputs");

  aicf::cuda::AttrPack pack{};
  std::vector<std::string> key_storage;
  std::vector<aicf::cuda::AttrKV> kv_storage;
  const void* attr_ptr = nullptr;

  if (attrs && attrs.size() > 0) {
    build_attr_pack_v0(attrs, key_storage, kv_storage, pack);
    attr_ptr = &pack;
  }

  // v0.1: stream = nullptr (backend decides)
  const cudaStream_t stream = nullptr;

  const aicf::Status st = aicf::cuda::dispatch_v0(
      kind,
      in_descs.data(),  (int)in_descs.size(),
      out_descs.data(), (int)out_descs.size(),
      attr_ptr,
      stream);

  if (!aicf::ok(st)) {
    throw std::runtime_error(
      std::string("aicf_cuda.op_call failed: kind=")
      + std::to_string((int)kind)
      + " status=" + aicf::status_to_string(st));
  }
}

PYBIND11_MODULE(aicf_cuda, m) {
  m.doc() = "AICF CUDA unified bindings (Plan A): op_call";

  ensure_kernels_registered_once();

  // expose OpKind enum
  py::enum_<aicf::cuda::OpKind>(m, "OpKind")
      .value("EltwiseAdd",  aicf::cuda::OpKind::EltwiseAdd)
      .value("EltwiseRelu", aicf::cuda::OpKind::EltwiseRelu)
      .value("Gemm",        aicf::cuda::OpKind::Gemm)
      .export_values();

  // op_call(kind: OpKind, inputs, outputs, attrs={})
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

  // optional compatibility overload: op_call(kind: int, ...)
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
}
