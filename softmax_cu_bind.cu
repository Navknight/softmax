// Benchmark binding for the homework submission. Includes softmax.cu
// UNMODIFIED and exposes its 5-kernel chain to Python so paper_bench.py can
// time it with the exact same protocol as every library. softmax.cu's main()
// comes along for the ride as an unused symbol in the shared library.
#include <torch/extension.h>
#include "softmax.cu"

torch::Tensor softmax_cu(torch::Tensor x, int blocks) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 1, "x must be 1-D");
    x = x.contiguous();
    int n = (int)x.numel();

    auto y = torch::empty_like(x);
    auto opts = x.options();
    auto partial = torch::empty({blocks}, opts);
    auto gmax = torch::empty({1}, opts);
    auto gsum = torch::empty({1}, opts);

    float *xp = x.data_ptr<float>();
    float *yp = y.data_ptr<float>();
    float *pp = partial.data_ptr<float>();
    float *mp = gmax.data_ptr<float>();
    float *sp = gsum.data_ptr<float>();

    k_max_partial<<<blocks, 256>>>(xp, n, pp);
    k_max_final  <<<1, 256>>>(pp, blocks, mp);
    k_sum_partial<<<blocks, 256>>>(xp, n, mp, pp);
    k_sum_final  <<<1, 256>>>(pp, blocks, sp);
    k_norm       <<<blocks, 256>>>(xp, yp, n, mp, sp);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_cu", &softmax_cu,
          "3-pass softmax from softmax.cu (homework submission kernels)",
          py::arg("x"), py::arg("blocks") = 320);
}
