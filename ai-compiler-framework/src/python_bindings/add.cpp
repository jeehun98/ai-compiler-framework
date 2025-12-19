#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "common.hpp"
#include <aicf/backends/cuda/ops/add/api.hpp>

namespace py = pybind11;

PYBIND11_MODULE(add, m) {
    m.doc() = "AICF CUDA add ops";

    m.def("add_f32", [](const torch::Tensor& a,
                        const torch::Tensor& b,
                        torch::Tensor& out) -> bool {
        if (!is_cuda_f32_contig(a) || !is_cuda_f32_contig(b) || !is_cuda_f32_contig(out)) return false;
        if (a.numel() != b.numel() || out.numel() != a.numel()) return false;

        const int N = (int)a.numel();
        auto st = aicf::cuda::add_f32(
            (const float*)a.data_ptr<float>(),
            (const float*)b.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            N, default_stream());

        return status_ok(st);
    }, py::arg("a"), py::arg("b"), py::arg("out"));
}
