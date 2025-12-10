#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "tensor.h"
#include "layers.h"
#include <vector>
#include <string>
#include <memory>

namespace py = pybind11;

using TensorF = Tensor<float>;
using TensorPtr = std::shared_ptr<TensorF>;
 
// 根据输入推断输出的Tensor
static TensorPtr make_tensor_like(const TensorPtr& src, const std::vector<int>& new_shape){
    Device dev = src->is_gpu() ? Device::GPU : Device::CPU;
    return std::make_shared<TensorF>(new_shape, dev);
}

static TensorPtr sigmoid_forward(const TensorPtr& input){
    auto in_shape = input->get_shape();
    TensorPtr output = make_tensor_like(input, in_shape);
    int totalnum = 1;
    for (auto i : in_shape){
        totalnum *= i;
    }
    sigmoid_gpu(input->data(), output->data(), totalnum);
    return output;
}

static TensorPtr sigmoid_backward(const TensorPtr& grad_output, const TensorPtr& output){
    auto in_shape = output->get_shape();
    TensorPtr grad_input = make_tensor_like(output, in_shape);
    int totalnum = 1;
    for (auto i : in_shape){
        totalnum *= i;
    }    
    sigmoid_gpu_backward( grad_input->data(), grad_output->data(), output->data(), totalnum);
    return grad_input;
}

static TensorPtr relu_forward(const TensorPtr& input){
    auto in_shape = input->get_shape();
    TensorPtr output = make_tensor_like(input, in_shape);
    int totalnum = 1;
    for (auto i : in_shape){
        totalnum *= i;
    }
    relu_gpu(input->data(), output->data(), totalnum);
    return output;
}

static TensorPtr relu_backward(const TensorPtr& grad_output, const TensorPtr& output){
    auto in_shape = output->get_shape();
    TensorPtr grad_input = make_tensor_like(output, in_shape);
    int totalnum = 1;
    for (auto i : in_shape){
        totalnum *= i;
    }
    relu_gpu_backward(grad_input->data(), grad_output->data(), output->data(), totalnum);
    return grad_input;
}

static TensorPtr fc_forward(const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias){
    //output(b, o) = input(b, i) * weight(i, o)   
    auto in_shape = input->get_shape();
    int batch = in_shape[0];
    int out_features = weight->get_shape()[1];
    std::vector<int> out_shape = {batch, out_features};
    TensorPtr output = make_tensor_like(input, out_shape);
    //void forward_fc(float* input, float* output, float* weight, float* bias,
    //    int batch_size, int out_features, int in_features){
    forward_fc(input->data(), output->data(), weight->data(), bias->data(), batch, out_features, in_shape[1]);
    return output;
}

static void fc_backward(const TensorPtr& grad_output, const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias,
                        const TensorPtr& grad_input, const TensorPtr& grad_weight, const TensorPtr& grad_bias){
    //void backward_fc(float* input, float* weight, float* bias,
    // int batch_size, int out_features, int in_features,
    // float* grad_input, float* grad_output, float* grad_weight, float* grad_bias)
    int batch_size = input->get_shape()[0];
    int out_features = weight->get_shape()[1];
    int in_features = weight->get_shape()[0];
    backward_fc(input->data(), weight->data(), bias->data(), batch_size, out_features, in_features,
                grad_input->data(), grad_output->data(), grad_weight->data(), grad_bias->data());
}

static TensorPtr conv2d_forward(const TensorPtr& input, const TensorPtr& filter){
    // output(batch_size, out_channels, height, width)
    // filter(out_channels, in_channels, 3, 3)
    // input(batch_size, in_channels, height, width)

    int batch_size = input->get_shape()[0];
    int out_channels = filter->get_shape()[0];
    int in_channels = input->get_shape()[1];
    int height = input->get_shape()[2];
    int width = input->get_shape()[3];
    std::vector<int> out_shape = {batch_size, out_channels, height, width};
    TensorPtr output = make_tensor_like(input, out_shape);
    // void forward_conv2d(float* input, float* output, float* filter,
    //             int batch_size, int out_channels, int in_channels, int height, int width,
    //             cudaStream_t stream){
    forward_conv2d(input->data(), output->data(), filter->data(),
                   batch_size, out_channels, in_channels, height, width, 0);
    return output;
}

static void conv2d_backward(const TensorPtr& grad_output, const TensorPtr& input, const TensorPtr& filter,
                            const TensorPtr& grad_input, const TensorPtr& grad_filter){
    // void backward_conv2d(float* input, float* filter,
    // int batch_size, int out_channels, int in_channels, int height, int width,
    // float* grad_input, float* grad_output, float* grad_filter,
    // cudaStream_t stream){
    int batch_size = input->get_shape()[0];
    int out_channels = filter->get_shape()[0];
    int in_channels = input->get_shape()[1];
    int height = input->get_shape()[2];
    int width = input->get_shape()[3];   
    backward_conv2d(input->data(), filter->data(),batch_size, out_channels, in_channels, height, width,
                    grad_input->data(), grad_output->data(), grad_filter->data(), 0);
}

static TensorPtr max_pool2d_forward(const TensorPtr& input){
    // void forward_maxpool(const float* input, float* output, float* mask,
    // int batch_size, int in_channels, int in_h, int in_w,
    // int out_h, int out_w, cudaStream_t stream){
    int batch_size = input->get_shape()[0];
    int in_channels = input->get_shape()[1];
    int in_h = input->get_shape()[2];
    int in_w = input->get_shape()[3];
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    std::vector<int> out_shape = {batch_size, in_channels, out_h, out_w};
    TensorPtr output = make_tensor_like(input, out_shape);
    TensorPtr mask = make_tensor_like(input, out_shape);
    forward_maxpool(input->data(), output->data(), mask->data(),
                    batch_size, in_channels, in_h, in_w, out_h, out_w, 0);
    return output;
}

static TensorPtr max_pool2d_forward_mask(const TensorPtr& input){
    int batch_size = input->get_shape()[0];
    int in_channels = input->get_shape()[1];
    int in_h = input->get_shape()[2];
    int in_w = input->get_shape()[3];
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    std::vector<int> out_shape = {batch_size, in_channels, out_h, out_w};
    TensorPtr output = make_tensor_like(input, out_shape);
    TensorPtr mask = make_tensor_like(input, out_shape);
    forward_maxpool(input->data(), output->data(), mask->data(),
                    batch_size, in_channels, in_h, in_w, out_h, out_w, 0);
    return mask;    
}

static void max_pool2d_backward(const TensorPtr& grad_output, const TensorPtr& mask, 
                                    const TensorPtr& input, const TensorPtr& grad_input){
    // void backward_maxpool(const float* grad_output, const float* mask, float* grad_input,
    // int batch_size, int in_channels, int in_h, int in_w,
    // int out_h, int out_w, cudaStream_t stream){                     
    int batch_size = input->get_shape()[0];
    int in_channels = input->get_shape()[1];
    int in_h = input->get_shape()[2];
    int in_w = input->get_shape()[3];
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    backward_maxpool(grad_output->data(), mask->data(), grad_input->data(),
                     batch_size, in_channels, in_h, in_w, out_h, out_w, 0);
}

static TensorPtr softmax_forward(const TensorPtr& input){
    // void forward_softmax(const float* input, float* output,
    // int batch_size ,int num_classes, cudaStream_t stream){
    int batch_size = input->get_shape()[0];
    int num_classes = input->get_shape()[1];
    TensorPtr output = make_tensor_like(input, input->get_shape());
    forward_softmax(input->data(), output->data(), batch_size, num_classes, 0);
    return output;
}

static float cross_entropy_forward(const TensorPtr& input, const TensorPtr& labels){
    // void forward_cross_entropy(const float* input, const int* labels, float* loss,
    //     int batch_size, int num_classes, cudaStream_t stream){

    //input shape (batch_size, num_classes)
    //labels shape (batch_size)
    int batch_size = input->get_shape()[0];
    int num_classes = input->get_shape()[1];
    float loss = 0.0f;
    forward_cross_entropy(input->data(), labels->data(), &loss, batch_size, num_classes, 0);
    return loss;
}

static void cross_entropy_backward(const TensorPtr& input, const TensorPtr& labels, TensorPtr& grad_input){
    // void backward_cross_entropy(const float* softmax_output, const float* labels,
    // int batch_size, int num_classes, float* grad_output, cudaStream_t stream)
    int batch_size = input->get_shape()[0];
    int num_classes = input->get_shape()[1];
    backward_cross_entropy(input->data(), labels->data(), batch_size, num_classes, grad_input->data(), 0);
}

static void sgd_step(std::vector<TensorPtr>& params, 
                     std::vector<TensorPtr>& grads, 
                     std::vector<TensorPtr>& velocities,
                     float lr, float momentum, float weight_decay) {
    if (params.size() != grads.size()) {
        throw std::runtime_error("params and grads must have the same size");
    }
    for (size_t i = 0; i < params.size(); ++i) {
        if (params[i]->is_cpu() || grads[i]->is_cpu()) {
             throw std::runtime_error("SGD step requires GPU tensors");
        }
        if (momentum > 0 && !velocities.empty()) {
            if (velocities[i]->is_cpu()) {
                throw std::runtime_error("SGD step requires GPU velocity tensors");
            }
        }

        int size = params[i]->get_size();
        float* v_ptr = (momentum > 0 && !velocities.empty()) ? velocities[i]->data() : nullptr;
        
        // Debug print
        // std::cout << "SGD Step: param=" << params[i]->data() 
        //           << " grad=" << grads[i]->data() 
        //           << " v=" << v_ptr 
        //           << " size=" << size 
        //           << " lr=" << lr << std::endl;

        sgd_update_gpu(params[i]->data(), grads[i]->data(), v_ptr, lr, momentum, weight_decay, size, 0);
    }
    cudaDeviceSynchronize();
}

static void adam_step(std::vector<TensorPtr>& params, 
                      std::vector<TensorPtr>& grads, 
                      std::vector<TensorPtr>& ms,
                      std::vector<TensorPtr>& vs,
                      float lr, float beta1, float beta2, float eps, float weight_decay, int t) {
    if (params.size() != grads.size()) {
        throw std::runtime_error("params and grads must have the same size");
    }
    for (size_t i = 0; i < params.size(); ++i) {
        int size = params[i]->get_size();
        adam_update_gpu(params[i]->data(), grads[i]->data(), ms[i]->data(), vs[i]->data(),
                        lr, beta1, beta2, eps, weight_decay, t, size, 0);
    }
}

// Element-wise bindings
static TensorPtr eltwise_add_op(const TensorPtr& a, const TensorPtr& b) {
    TensorPtr out = make_tensor_like(a, a->get_shape());
    eltwise_add(a->data(), b->data(), out->data(), a->get_size());
    return out;
}

static TensorPtr eltwise_sub_op(const TensorPtr& a, const TensorPtr& b) {
    TensorPtr out = make_tensor_like(a, a->get_shape());
    eltwise_sub(a->data(), b->data(), out->data(), a->get_size());
    return out;
}

static TensorPtr eltwise_mul_op(const TensorPtr& a, const TensorPtr& b) {
    TensorPtr out = make_tensor_like(a, a->get_shape());
    eltwise_mul(a->data(), b->data(), out->data(), a->get_size());
    return out;
}

static TensorPtr eltwise_div_op(const TensorPtr& a, const TensorPtr& b) {
    TensorPtr out = make_tensor_like(a, a->get_shape());
    eltwise_div(a->data(), b->data(), out->data(), a->get_size());
    return out;
}

static TensorPtr eltwise_pow_op(const TensorPtr& a, const TensorPtr& b) {
    TensorPtr out = make_tensor_like(a, a->get_shape());
    eltwise_pow(a->data(), b->data(), out->data(), a->get_size());
    return out;
}

// Scalar bindings
static TensorPtr scalar_add_op(const TensorPtr& a, float val) {
    TensorPtr out = make_tensor_like(a, a->get_shape());
    scalar_add(a->data(), val, out->data(), a->get_size());
    return out;
}

static TensorPtr scalar_mul_op(const TensorPtr& a, float val) {
    TensorPtr out = make_tensor_like(a, a->get_shape());
    scalar_mul(a->data(), val, out->data(), a->get_size());
    return out;
}

static TensorPtr scalar_div_op(const TensorPtr& a, float val) {
    TensorPtr out = make_tensor_like(a, a->get_shape());
    scalar_div(a->data(), val, out->data(), a->get_size());
    return out;
}

static TensorPtr scalar_pow_op(const TensorPtr& a, float val) {
    TensorPtr out = make_tensor_like(a, a->get_shape());
    scalar_pow(a->data(), val, out->data(), a->get_size());
    return out;
}

static void empty_cache(){
    MemoryPool::instance().clear();
}

PYBIND11_MODULE(py_tensor, m) {
    m.doc() = "py_tensor plugin";
    m.def("empty_cache", &empty_cache, "Clear the memory pool");

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
        }, py::arg("array"), py::arg("device") = "cpu");

    m.def("relu_forward", &relu_forward, py::arg("input"));
    m.def("relu_backward", &relu_backward, py::arg("output_grad"), py::arg("output"));
    m.def("sigmoid_forward", &sigmoid_forward, py::arg("input"));
    m.def("sigmoid_backward", &sigmoid_backward, py::arg("output_grad"), py::arg("output"));
    m.def("fc_forward", &fc_forward, py::arg("input"), py::arg("weight"), py::arg("bias"));
    m.def("fc_backward", &fc_backward, py::arg("output_grad"), py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("input_grad"), py::arg("weight_grad"), py::arg("bias_grad"));
    m.def("conv2d_forward", &conv2d_forward, py::arg("input"), py::arg("filter"));
    m.def("conv2d_backward", &conv2d_backward, py::arg("output_grad"), py::arg("input"), py::arg("filter"),
          py::arg("input_grad"), py::arg("filter_grad"));
    m.def("max_pool2d_forward", &max_pool2d_forward, py::arg("input"));
    m.def("max_pool2d_forward_mask", &max_pool2d_forward_mask, py::arg("input"));
    m.def("max_pool2d_backward", &max_pool2d_backward, py::arg("output_grad"), py::arg("mask"), py::arg("input"), py::arg("input_grad"));
    m.def("softmax_forward", &softmax_forward, py::arg("input"));
    m.def("cross_entropy_forward", &cross_entropy_forward, py::arg("input"), py::arg("labels"));
    m.def("cross_entropy_backward", &cross_entropy_backward, py::arg("input"), py::arg("labels"), py::arg("input_grad"));
    m.def("sgd_step", &sgd_step, py::arg("params"), py::arg("grads"), py::arg("velocities"),
          py::arg("lr"), py::arg("momentum"), py::arg("weight_decay"));
    m.def("adam_step", &adam_step, py::arg("params"), py::arg("grads"), py::arg("ms"), py::arg("vs"),
          py::arg("lr"), py::arg("beta1"), py::arg("beta2"), py::arg("eps"), py::arg("weight_decay"), py::arg("t"));

    m.def("eltwise_add", &eltwise_add_op);
    m.def("eltwise_sub", &eltwise_sub_op);
    m.def("eltwise_mul", &eltwise_mul_op);
    m.def("eltwise_div", &eltwise_div_op);
    m.def("eltwise_pow", &eltwise_pow_op);
    
    m.def("scalar_add", &scalar_add_op);
    m.def("scalar_mul", &scalar_mul_op);
    m.def("scalar_div", &scalar_div_op);
    m.def("scalar_pow", &scalar_pow_op);
}