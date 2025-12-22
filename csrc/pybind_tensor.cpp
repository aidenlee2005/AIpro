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
    forward_fc(input->data(), output->data(), weight->data(), bias->data(), batch, out_features, in_shape[1]);
    return output;
}

static std::tuple<TensorPtr, TensorPtr, TensorPtr> fc_backward(const TensorPtr& grad_output, const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias){
    int batch_size = input->get_shape()[0];
    int out_features = weight->get_shape()[1];
    int in_features = weight->get_shape()[0];
    
    TensorPtr grad_input = make_tensor_like(input, input->get_shape());
    TensorPtr grad_weight = make_tensor_like(weight, weight->get_shape());
    TensorPtr grad_bias = make_tensor_like(bias, bias->get_shape());
    
    backward_fc(input->data(), weight->data(), bias->data(), batch_size, out_features, in_features,
                grad_input->data(), grad_output->data(), grad_weight->data(), grad_bias->data());
                
    return std::make_tuple(grad_input, grad_weight, grad_bias);
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

static std::tuple<TensorPtr, TensorPtr> conv2d_backward(const TensorPtr& grad_output, const TensorPtr& input, const TensorPtr& filter){
    int batch_size = input->get_shape()[0];
    int out_channels = filter->get_shape()[0];
    int in_channels = input->get_shape()[1];
    int height = input->get_shape()[2];
    int width = input->get_shape()[3];   
    
    TensorPtr grad_input = make_tensor_like(input, input->get_shape());
    TensorPtr grad_filter = make_tensor_like(filter, filter->get_shape());
    
    backward_conv2d(input->data(), filter->data(),batch_size, out_channels, in_channels, height, width,
                    grad_input->data(), grad_output->data(), grad_filter->data(), 0);
                    
    return std::make_tuple(grad_input, grad_filter);
}

// Fused Conv2D + ReLU forward
static TensorPtr conv2d_relu_forward(const TensorPtr& input, const TensorPtr& filter, 
                                    int kernel_size = 3, int stride = 1, int padding = 1){
    int batch_size = input->get_shape()[0];
    int out_channels = filter->get_shape()[0];
    int in_channels = input->get_shape()[1];
    int height = input->get_shape()[2];
    int width = input->get_shape()[3];
    
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    std::vector<int> out_shape = {batch_size, out_channels, out_height, out_width};
    
    TensorPtr output = make_tensor_like(input, out_shape);
    
    // Optional bias
    const float* bias_ptr = nullptr;
    // We can add bias argument to this function, but for now let's keep it simple or add it.
    // Let's assume no bias for now in this wrapper, or update wrapper signature.
    // But wait, I need to update the wrapper signature to support bias.
    
    conv2d_relu_forward_gpu(input->data(), filter->data(), nullptr, output->data(),
                           batch_size, out_channels, in_channels, height, width,
                           kernel_size, stride, padding, 0);
    return output;
}

// Fused Conv2D + ReLU forward with bias
static TensorPtr conv2d_relu_forward_bias(const TensorPtr& input, const TensorPtr& filter, const TensorPtr& bias,
                                         int kernel_size = 3, int stride = 1, int padding = 1){
    int batch_size = input->get_shape()[0];
    int out_channels = filter->get_shape()[0];
    int in_channels = input->get_shape()[1];
    int height = input->get_shape()[2];
    int width = input->get_shape()[3];
    
    int out_height = (height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (width + 2 * padding - kernel_size) / stride + 1;
    std::vector<int> out_shape = {batch_size, out_channels, out_height, out_width};
    
    TensorPtr output = make_tensor_like(input, out_shape);
    
    conv2d_relu_forward_gpu(input->data(), filter->data(), bias->data(), output->data(),
                           batch_size, out_channels, in_channels, height, width,
                           kernel_size, stride, padding, 0);
    return output;
}

// Fused Conv2D + ReLU backward
static std::tuple<TensorPtr, TensorPtr> conv2d_relu_backward(const TensorPtr& grad_output, const TensorPtr& output,
                                                            const TensorPtr& input, const TensorPtr& filter,
                                                            int kernel_size = 3, int stride = 1, int padding = 1){
    int batch_size = input->get_shape()[0];
    int out_channels = filter->get_shape()[0];
    int in_channels = input->get_shape()[1];
    int height = input->get_shape()[2];
    int width = input->get_shape()[3];
    
    TensorPtr grad_input = make_tensor_like(input, input->get_shape());
    TensorPtr grad_filter = make_tensor_like(filter, filter->get_shape());
    
    conv2d_relu_backward_gpu(grad_output->data(), output->data(), input->data(), filter->data(),
                            grad_input->data(), grad_filter->data(), nullptr,
                            batch_size, out_channels, in_channels, height, width,
                            kernel_size, stride, padding, 0);
                            
    return std::make_tuple(grad_input, grad_filter);
}

// Fused Conv2D + ReLU backward with bias
static std::tuple<TensorPtr, TensorPtr, TensorPtr> conv2d_relu_backward_bias(const TensorPtr& grad_output, const TensorPtr& output,
                                                            const TensorPtr& input, const TensorPtr& filter, const TensorPtr& bias,
                                                            int kernel_size = 3, int stride = 1, int padding = 1){
    int batch_size = input->get_shape()[0];
    int out_channels = filter->get_shape()[0];
    int in_channels = input->get_shape()[1];
    int height = input->get_shape()[2];
    int width = input->get_shape()[3];
    
    TensorPtr grad_input = make_tensor_like(input, input->get_shape());
    TensorPtr grad_filter = make_tensor_like(filter, filter->get_shape());
    TensorPtr grad_bias = make_tensor_like(bias, bias->get_shape());
    
    conv2d_relu_backward_gpu(grad_output->data(), output->data(), input->data(), filter->data(),
                            grad_input->data(), grad_filter->data(), grad_bias->data(),
                            batch_size, out_channels, in_channels, height, width,
                            kernel_size, stride, padding, 0);
                            
    return std::make_tuple(grad_input, grad_filter, grad_bias);
}

static TensorPtr max_pool2d_forward(const TensorPtr& input){
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

static TensorPtr max_pool2d_backward(const TensorPtr& grad_output, const TensorPtr& mask, 
                                const TensorPtr& input){
    int batch_size = input->get_shape()[0];
    int in_channels = input->get_shape()[1];
    int in_h = input->get_shape()[2];
    int in_w = input->get_shape()[3];
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    
    TensorPtr grad_input = make_tensor_like(input, input->get_shape());
    
    backward_maxpool(grad_output->data(), mask->data(), grad_input->data(),
                     batch_size, in_channels, in_h, in_w, out_h, out_w, 0);
                     
    return grad_input;
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

static TensorPtr cross_entropy_backward(const TensorPtr& input, const TensorPtr& labels){
    // void backward_cross_entropy(const float* softmax_output, const float* labels,
    // int batch_size, int num_classes, float* grad_output, cudaStream_t stream)
    int batch_size = input->get_shape()[0];
    int num_classes = input->get_shape()[1];
    
    TensorPtr grad_input = make_tensor_like(input, input->get_shape());
    
    backward_cross_entropy(input->data(), labels->data(), batch_size, num_classes, grad_input->data(), 0);
    
    return grad_input;
}

static void sgd_step(std::vector<TensorPtr>& params, 
                     std::vector<TensorPtr>& grads, 
                     std::vector<TensorPtr>& velocities,
                     float lr, float momentum, float weight_decay) {
    if (params.size() != grads.size()) {
        throw std::runtime_error("params and grads must have the same size");
    }
    
    // Create CUDA stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
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
        
        sgd_update_gpu(params[i]->data(), grads[i]->data(), v_ptr, lr, momentum, weight_decay, size, stream);
    }
    
    // Synchronize stream instead of device
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

static void batch_sgd_step(std::vector<TensorPtr>& params, 
                           std::vector<TensorPtr>& grads, 
                           std::vector<TensorPtr>& velocities,
                           std::vector<int>& param_sizes,
                           float lr, float momentum, float weight_decay) {
    // Create CUDA stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Calculate total size and offsets
    size_t total_size = 0;
    std::vector<size_t> offsets;
    for (int size : param_sizes) {
        offsets.push_back(total_size);
        total_size += size;
    }
    
    // Allocate contiguous memory for all parameters and gradients
    TensorPtr all_params = std::make_shared<TensorF>(std::vector<int>{(int)total_size}, Device::GPU);
    TensorPtr all_grads = std::make_shared<TensorF>(std::vector<int>{(int)total_size}, Device::GPU);
    TensorPtr all_velocities = nullptr;
    if (momentum > 0) {
        all_velocities = std::make_shared<TensorF>(std::vector<int>{(int)total_size}, Device::GPU);
    }
    
    // Copy data to contiguous arrays
    size_t current_offset = 0;
    for (size_t i = 0; i < params.size(); ++i) {
        int size = param_sizes[i];
        cudaMemcpyAsync(all_params->data() + current_offset, params[i]->data(), 
                       size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(all_grads->data() + current_offset, grads[i]->data(), 
                       size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (momentum > 0) {
            cudaMemcpyAsync(all_velocities->data() + current_offset, velocities[i]->data(), 
                           size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }
        current_offset += size;
    }
    
    // Launch batched kernel
    batch_sgd_update_gpu(all_params->data(), all_grads->data(), 
                         all_velocities ? all_velocities->data() : nullptr,
                         lr, momentum, weight_decay, total_size, stream);
    
    // Copy results back
    current_offset = 0;
    for (size_t i = 0; i < params.size(); ++i) {
        int size = param_sizes[i];
        cudaMemcpyAsync(params[i]->data(), all_params->data() + current_offset, 
                       size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        if (momentum > 0) {
            cudaMemcpyAsync(velocities[i]->data(), all_velocities->data() + current_offset, 
                           size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }
        current_offset += size;
    }
    
    // Synchronize stream
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

static void adam_step(std::vector<TensorPtr>& params, 
                      std::vector<TensorPtr>& grads, 
                      std::vector<TensorPtr>& ms,
                      std::vector<TensorPtr>& vs,
                      float lr, float beta1, float beta2, float eps, float weight_decay, int t) {
    if (params.size() != grads.size()) {
        throw std::runtime_error("params and grads must have the same size");
    }
    
    // Create CUDA stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    for (size_t i = 0; i < params.size(); ++i) {
        int size = params[i]->get_size();
        adam_update_gpu(params[i]->data(), grads[i]->data(), ms[i]->data(), vs[i]->data(),
                        lr, beta1, beta2, eps, weight_decay, t, size, stream);
    }
    
    // Synchronize stream instead of device
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}

// Helper for broadcasting
static bool compute_broadcast_strides(const std::vector<int>& shape_a, const std::vector<int>& shape_b,
                                      std::vector<int>& out_shape,
                                      TensorStrides& out_strides,
                                      TensorStrides& a_strides,
                                      TensorStrides& b_strides,
                                      int& ndim) {
    int ndim_a = shape_a.size();
    int ndim_b = shape_b.size();
    ndim = std::max(ndim_a, ndim_b);
    
    if (ndim > MAX_DIMS) return false; // Too many dims

    out_shape.resize(ndim);
    
    // Compute output shape
    for (int i = 0; i < ndim; ++i) {
        int dim_a = (i < ndim - ndim_a) ? 1 : shape_a[i - (ndim - ndim_a)];
        int dim_b = (i < ndim - ndim_b) ? 1 : shape_b[i - (ndim - ndim_b)];
        
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false; // Incompatible shapes
        }
        out_shape[i] = std::max(dim_a, dim_b);
    }

    // Compute strides
    // Real strides for A
    std::vector<int> real_strides_a(ndim_a);
    int stride = 1;
    for (int i = ndim_a - 1; i >= 0; --i) {
        real_strides_a[i] = stride;
        stride *= shape_a[i];
    }
    
    // Real strides for B
    std::vector<int> real_strides_b(ndim_b);
    stride = 1;
    for (int i = ndim_b - 1; i >= 0; --i) {
        real_strides_b[i] = stride;
        stride *= shape_b[i];
    }

    // Virtual strides relative to out_shape
    // Out strides
    stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        out_strides.data[i] = stride;
        stride *= out_shape[i];
    }

    for (int i = 0; i < ndim; ++i) {
        // Map out dim i to A dim
        int offset_a = i - (ndim - ndim_a);
        if (offset_a >= 0) {
            int dim_a = shape_a[offset_a];
            a_strides.data[i] = (dim_a == 1) ? 0 : real_strides_a[offset_a];
        } else {
            a_strides.data[i] = 0; // Broadcast (prepend 1)
        }

        // Map out dim i to B dim
        int offset_b = i - (ndim - ndim_b);
        if (offset_b >= 0) {
            int dim_b = shape_b[offset_b];
            b_strides.data[i] = (dim_b == 1) ? 0 : real_strides_b[offset_b];
        } else {
            b_strides.data[i] = 0; // Broadcast (prepend 1)
        }
    }
    
    return true;
}

// Element-wise bindings
static TensorPtr eltwise_add_op(const TensorPtr& a, const TensorPtr& b) {
    if (a->get_shape() == b->get_shape()) {
        TensorPtr out = make_tensor_like(a, a->get_shape());
        eltwise_add(a->data(), b->data(), out->data(), a->get_size());
        return out;
    } else {
        std::vector<int> out_shape;
        TensorStrides out_s, a_s, b_s;
        int ndim;
        if (compute_broadcast_strides(a->get_shape(), b->get_shape(), out_shape, out_s, a_s, b_s, ndim)) {
            TensorPtr out = make_tensor_like(a, out_shape); // Use a's device
            eltwise_add_broadcast(a->data(), b->data(), out->data(), out->get_size(), ndim, out_s, a_s, b_s);
            return out;
        } else {
             throw std::runtime_error("Incompatible shapes for broadcasting or too many dims");
        }
    }
}

static TensorPtr eltwise_sub_op(const TensorPtr& a, const TensorPtr& b) {
    if (a->get_shape() == b->get_shape()) {
        TensorPtr out = make_tensor_like(a, a->get_shape());
        eltwise_sub(a->data(), b->data(), out->data(), a->get_size());
        return out;
    } else {
        std::vector<int> out_shape;
        TensorStrides out_s, a_s, b_s;
        int ndim;
        if (compute_broadcast_strides(a->get_shape(), b->get_shape(), out_shape, out_s, a_s, b_s, ndim)) {
            TensorPtr out = make_tensor_like(a, out_shape);
            eltwise_sub_broadcast(a->data(), b->data(), out->data(), out->get_size(), ndim, out_s, a_s, b_s);
            return out;
        } else {
             throw std::runtime_error("Incompatible shapes for broadcasting or too many dims");
        }
    }
}

static TensorPtr eltwise_mul_op(const TensorPtr& a, const TensorPtr& b) {
    if (a->get_shape() == b->get_shape()) {
        TensorPtr out = make_tensor_like(a, a->get_shape());
        eltwise_mul(a->data(), b->data(), out->data(), a->get_size());
        return out;
    } else {
        std::vector<int> out_shape;
        TensorStrides out_s, a_s, b_s;
        int ndim;
        if (compute_broadcast_strides(a->get_shape(), b->get_shape(), out_shape, out_s, a_s, b_s, ndim)) {
            TensorPtr out = make_tensor_like(a, out_shape);
            eltwise_mul_broadcast(a->data(), b->data(), out->data(), out->get_size(), ndim, out_s, a_s, b_s);
            return out;
        } else {
             throw std::runtime_error("Incompatible shapes for broadcasting or too many dims");
        }
    }
}

static TensorPtr eltwise_div_op(const TensorPtr& a, const TensorPtr& b) {
    if (a->get_shape() == b->get_shape()) {
        TensorPtr out = make_tensor_like(a, a->get_shape());
        eltwise_div(a->data(), b->data(), out->data(), a->get_size());
        return out;
    } else {
        std::vector<int> out_shape;
        TensorStrides out_s, a_s, b_s;
        int ndim;
        if (compute_broadcast_strides(a->get_shape(), b->get_shape(), out_shape, out_s, a_s, b_s, ndim)) {
            TensorPtr out = make_tensor_like(a, out_shape);
            eltwise_div_broadcast(a->data(), b->data(), out->data(), out->get_size(), ndim, out_s, a_s, b_s);
            return out;
        } else {
             throw std::runtime_error("Incompatible shapes for broadcasting or too many dims");
        }
    }
}

static TensorPtr eltwise_pow_op(const TensorPtr& a, const TensorPtr& b) {
    if (a->get_shape() == b->get_shape()) {
        TensorPtr out = make_tensor_like(a, a->get_shape());
        eltwise_pow(a->data(), b->data(), out->data(), a->get_size());
        return out;
    } else {
        std::vector<int> out_shape;
        TensorStrides out_s, a_s, b_s;
        int ndim;
        if (compute_broadcast_strides(a->get_shape(), b->get_shape(), out_shape, out_s, a_s, b_s, ndim)) {
            TensorPtr out = make_tensor_like(a, out_shape);
            eltwise_pow_broadcast(a->data(), b->data(), out->data(), out->get_size(), ndim, out_s, a_s, b_s);
            return out;
        } else {
             throw std::runtime_error("Incompatible shapes for broadcasting or too many dims");
        }
    }
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

// BatchNorm Wrappers

static std::tuple<TensorPtr, TensorPtr, TensorPtr> batch_norm_forward_training_wrapper(
    const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias,
    const TensorPtr& running_mean, const TensorPtr& running_var,
    float momentum, float eps) {
    
    auto shape = input->get_shape();
    int batch_size = shape[0];
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];
    
    TensorPtr output = make_tensor_like(input, shape);
    TensorPtr save_mean = make_tensor_like(input, {channels});
    TensorPtr save_inv_std = make_tensor_like(input, {channels});
    
    batch_norm_forward_training(
        input->data(), output->data(),
        weight->data(), bias->data(),
        running_mean->data(), running_var->data(),
        save_mean->data(), save_inv_std->data(),
        batch_size, channels, height, width,
        momentum, eps);
        
    return std::make_tuple(output, save_mean, save_inv_std);
}

static TensorPtr batch_norm_forward_inference_wrapper(
    const TensorPtr& input, const TensorPtr& weight, const TensorPtr& bias,
    const TensorPtr& running_mean, const TensorPtr& running_var,
    float eps) {
    
    auto shape = input->get_shape();
    int batch_size = shape[0];
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];
    
    TensorPtr output = make_tensor_like(input, shape);
    
    batch_norm_forward_inference(
        input->data(), output->data(),
        weight->data(), bias->data(),
        running_mean->data(), running_var->data(),
        batch_size, channels, height, width,
        eps);
        
    return output;
}

static std::tuple<TensorPtr, TensorPtr, TensorPtr> batch_norm_backward_wrapper(
    const TensorPtr& grad_output, const TensorPtr& input,
    const TensorPtr& weight,
    const TensorPtr& save_mean, const TensorPtr& save_inv_std) {
    
    auto shape = input->get_shape();
    int batch_size = shape[0];
    int channels = shape[1];
    int height = shape[2];
    int width = shape[3];
    
    TensorPtr grad_input = make_tensor_like(input, shape);
    TensorPtr grad_weight = make_tensor_like(input, {channels});
    TensorPtr grad_bias = make_tensor_like(input, {channels});
    
    batch_norm_backward(
        grad_output->data(), input->data(), grad_input->data(),
        weight->data(), grad_weight->data(), grad_bias->data(),
        save_mean->data(), save_inv_std->data(),
        batch_size, channels, height, width);
        
    return std::make_tuple(grad_input, grad_weight, grad_bias);
}

static void empty_cache(){
    MemoryPool::instance().clear();
}

// Dropout Forward
// Returns {output, mask}
static std::tuple<TensorPtr, TensorPtr> dropout_forward_wrapper(const TensorPtr& input, float dropout_p, unsigned long long seed) {
    if (input->is_cpu()) {
        throw std::runtime_error("Dropout not implemented for CPU");
    }
    auto shape = input->get_shape();
    int size = input->get_size();
    
    TensorPtr output = make_tensor_like(input, shape);
    TensorPtr mask = make_tensor_like(input, shape);
    TensorPtr rand = make_tensor_like(input, shape); // Temporary Tensor
    
    // 1. Generate random numbers
    fill_random_uniform(rand->data(), size, seed);
    
    // 2. Compute Dropout
    float keep_prob = 1.0f - dropout_p;
    dropout_forward(input->data(), output->data(), mask->data(), rand->data(), size, keep_prob, 0);
    
    cudaDeviceSynchronize(); // Ensure kernels finish before rand is destroyed
    
    return std::make_tuple(output, mask);
}

// Dropout Backward
static TensorPtr dropout_backward_wrapper(const TensorPtr& grad_output, const TensorPtr& mask, float dropout_p) {
    if (grad_output->is_cpu()) {
        throw std::runtime_error("Dropout backward not implemented for CPU");
    }
    auto shape = grad_output->get_shape();
    int size = grad_output->get_size();
    TensorPtr grad_input = make_tensor_like(grad_output, shape);
    
    float keep_prob = 1.0f - dropout_p;
    dropout_backward(grad_output->data(), mask->data(), grad_input->data(), size, keep_prob, 0);
    
    cudaDeviceSynchronize();
    
    return grad_input;
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
    m.def("fc_backward", &fc_backward, py::arg("output_grad"), py::arg("input"), py::arg("weight"), py::arg("bias"));
    m.def("conv2d_forward", &conv2d_forward, py::arg("input"), py::arg("filter"));
    m.def("conv2d_backward", &conv2d_backward, py::arg("output_grad"), py::arg("input"), py::arg("filter"));
    m.def("conv2d_relu_forward", &conv2d_relu_forward, py::arg("input"), py::arg("filter"), 
          py::arg("kernel_size") = 3, py::arg("stride") = 1, py::arg("padding") = 1);
    m.def("conv2d_relu_forward_bias", &conv2d_relu_forward_bias, py::arg("input"), py::arg("filter"), py::arg("bias"),
          py::arg("kernel_size") = 3, py::arg("stride") = 1, py::arg("padding") = 1);
    m.def("conv2d_relu_backward", &conv2d_relu_backward, py::arg("output_grad"), py::arg("output"), py::arg("input"), py::arg("filter"),
          py::arg("kernel_size") = 3, py::arg("stride") = 1, py::arg("padding") = 1);
    m.def("conv2d_relu_backward_bias", &conv2d_relu_backward_bias, py::arg("output_grad"), py::arg("output"), py::arg("input"), py::arg("filter"), py::arg("bias"),
          py::arg("kernel_size") = 3, py::arg("stride") = 1, py::arg("padding") = 1);
    m.def("max_pool2d_forward", &max_pool2d_forward, py::arg("input"));
    m.def("max_pool2d_forward_mask", &max_pool2d_forward_mask, py::arg("input"));
    m.def("max_pool2d_backward", &max_pool2d_backward, py::arg("output_grad"), py::arg("mask"), py::arg("input"));
    m.def("softmax_forward", &softmax_forward, py::arg("input"));
    m.def("cross_entropy_forward", &cross_entropy_forward, py::arg("input"), py::arg("labels"));
    m.def("cross_entropy_backward", &cross_entropy_backward, py::arg("input"), py::arg("labels"));
    m.def("sgd_step", &sgd_step, py::arg("params"), py::arg("grads"), py::arg("velocities"),
          py::arg("lr"), py::arg("momentum"), py::arg("weight_decay"));
    m.def("batch_sgd_step", &batch_sgd_step, py::arg("params"), py::arg("grads"), py::arg("velocities"),
          py::arg("param_sizes"), py::arg("lr"), py::arg("momentum"), py::arg("weight_decay"));
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

    m.def("batch_norm_forward_training", &batch_norm_forward_training_wrapper);
    m.def("batch_norm_forward_inference", &batch_norm_forward_inference_wrapper);
    m.def("batch_norm_backward", &batch_norm_backward_wrapper);

    m.def("dropout_forward", &dropout_forward_wrapper);
    m.def("dropout_backward", &dropout_backward_wrapper);
}