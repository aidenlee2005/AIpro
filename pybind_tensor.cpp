#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tensor.h"
#include <vector>
#include <string>
#include <memory> // <- 新增

namespace py = pybind11;

using TensorF = Tensor<float>;
using TensorPtr = std::shared_ptr<TensorF>;

PYBIND11_MODULE(py_tensor, m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    // 使用 shared_ptr 作为 holder type
    py::class_<TensorF, TensorPtr>(m, "Tensor")
        .def(py::init([](std::vector<int> shape, std::string device_str){
            Device d = (device_str == "gpu" || device_str == "GPU") ? Device::GPU : Device::CPU;
            return std::make_shared<TensorF>(shape, d);
        }), py::arg("shape"), py::arg("device") = "cpu")
        .def("shape", &TensorF::get_shape)
        .def("size", &TensorF::get_size)
        .def("is_cpu", &TensorF::is_cpu)
        .def("is_gpu", &TensorF::is_gpu)
        .def("print", &TensorF::print)
        .def("to_numpy", [](TensorPtr t){
            std::vector<int> s = t->get_shape();
            size_t n = t->get_size();
            auto *host = new std::vector<float>(n);
            t->copy_to_host(host->data());

            std::vector<ssize_t> pyshape(s.begin(), s.end());
            std::vector<ssize_t> pystrides(s.size());
            ssize_t stride = sizeof(float);
            for (int i = (int)s.size() - 1; i >= 0; --i) {
                pystrides[i] = stride;
                stride *= pyshape[i];
            }

            py::capsule free_host(host, [](void *f){
                delete static_cast<std::vector<float>*>(f);
            });

            return py::array(py::buffer_info(
                host->data(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                pyshape.size(),
                pyshape,
                pystrides
            ), free_host);
        })
        .def_static("from_numpy", [](py::array_t<float, py::array::c_style | py::array::forcecast> arr, std::string device_str){
            py::buffer_info info = arr.request();
            std::vector<int> shape(info.shape.begin(), info.shape.end());
            auto t = std::make_shared<TensorF>(shape,
                (device_str == "gpu" || device_str == "GPU") ? Device::GPU : Device::CPU);
            float* src = static_cast<float*>(info.ptr);
            t->copy_from_host(src);
            return t;
        }, py::arg("array"), py::arg("device") = "cpu")
        ;
}