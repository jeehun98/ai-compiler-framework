#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "common.hpp"
#include <aicf/backends/cuda/ops/relu/api.hpp>

namespace py = pybind11;

PYBIND11_MODULE(relu, m) {
    m.doc() = "AICF CUDA relu ops";

    m.def("relu_f32", [](const torch::Tensor& in,
                         torch::Tensor& out) -> bool {
        if (!is_cuda_f32_contig(in) || !is_cuda_f32_contig(out)) return false;
        if (out.numel() != in.numel()) return false;

        const int N = (int)in.numel();
        auto st = aicf::cuda::relu_f32(
            (const float*)in.data_ptr<float>(),
            (float*)out.data_ptr<float>(),
            N, default_stream());

        return status_ok(st);
    }, py::arg("in"), py::arg("out"));
}
