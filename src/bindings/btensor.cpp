#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h> 
#include "core/tensor.hh"

namespace py = pybind11;

PYBIND11_MODULE(nanograd, m){
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .export_values();
    
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "tensor")
        .def_static("create_shape",
            py::overload_cast<const std::vector<int>&, DeviceType>(&Tensor::create),
            py::arg("shape"),
            py::arg("device") = DeviceType::CPU
        ).def_static("create_data",
            py::overload_cast<const std::vector<float>&, const std::vector<int>&, DeviceType>(&Tensor::create),
            py::arg("data"),
            py::arg("shape"),
            py::arg("device") = DeviceType::CPU
        ).def("shape", &Tensor::shape)
        .def("device", &Tensor::device)
        .def("numel", &Tensor::numel)
        .def("to", &Tensor::to)
        .def("zero_grad", &Tensor::zero_grad)
        .def("sum", &Tensor::sum)
        .def("data", [](const Tensor& self) {
            float* data_ptr = self.data();
            size_t numel = self.numel(); 
            
            return py::array_t<float>(
                {numel},        
                {},             
                data_ptr
            );
        })        
        .def("grad", [](const Tensor& self) -> py::array_t<float> {
            if (!self.grad()) {
                return py::array_t<float>(); 
            }
            float* grad_ptr = self.grad();
            size_t numel = self.numel();
            return py::array_t<float>(
                {numel},
                {},
                grad_ptr
            );
        })
        .def("backward", [](Tensor& self, const std::shared_ptr<Tensor>& upstream_grad){
            self.backward(upstream_grad);
        }, py::arg("upstream_grad") = nullptr)
        .def("__add__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b){
            return *a + b;
        })
        .def("__mul__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b){
            return *a * b;
        })
        .def("__truediv__", [](const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b){
            return *a / b;
        })
        .def("__getitem__",  [](const Tensor& self, const std::vector<int>& indices){
            return self[indices];
        })
        .def("__setitem__", [](Tensor& self, const std::vector<int>& indices, float value){
            self[indices] = value;
        });
}