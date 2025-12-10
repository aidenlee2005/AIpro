#ifndef LAYERS_H
#define LAYERS_H

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#ifndef CUDA_KERNEL_LOOP
#define CUDA_KERNEL_LOOP(i,n)\
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x)
#endif

enum class TransposeType {
    NoTranspose,
    Transpose,
};

void relu_gpu(float* in, float* out, int size);

void relu_gpu_backward(float* in_grad, float* out_grad, float* x, int size);

void sigmoid_gpu(float* in, float* out, int size);

void sigmoid_gpu_backward(float* in_grad, float* out_grad, float* out, int size);

void forward_fc(float* input, float* output, float* weight, float* bias,
    int batch_size, int out_features, int in_features);

void backward_fc(float* input, float* weight, float* bias,
    int batch_size, int out_features, int in_features,
    float* grad_input, float* grad_output, float* grad_weight, float* grad_bias);
    

//Task2 : Convolutional Layer

void forward_conv2d(float* input, float* output, float* filter,
                int batch_size, int out_channels, int in_channels, int height, int width,
                cudaStream_t stream);

void backward_conv2d(float* input, float* filter,
        int batch_size, int out_channels, int in_channels, int height, int width,
        float* grad_input, float* grad_output, float* grad_filter,
        cudaStream_t stream);
        
//Task 3: Max Pooling Layer

     
void forward_maxpool(const float* input, float* output, float* mask,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_h, int out_w, cudaStream_t stream);

void backward_maxpool(const float* grad_output, const float* mask, float* grad_input,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_h, int out_w, cudaStream_t stream);

//Task4: Softmax Layer

void forward_softmax(const float* input, float* output,
    int batch_size ,int num_classes, cudaStream_t stream);

//Task5: Cross Entropy Loss Layer

void forward_cross_entropy(const float* input, const float* labels, float* loss,
    int batch_size, int num_classes, cudaStream_t stream);

void backward_cross_entropy(const float* softmax_output, const float* labels,
    int batch_size, int num_classes, float* grad_output, cudaStream_t stream);

// SGD and Adam update functions
void sgd_update_gpu(float* param, const float* grad, float* velocity, 
                    float lr, float momentum, float weight_decay, int size, cudaStream_t stream);

void adam_update_gpu(float* param, const float* grad, float* m, float* v,
                     float lr, float beta1, float beta2, float eps, float weight_decay, 
                     int step, int size, cudaStream_t stream);

// Element-wise operations
void eltwise_add(const float* a, const float* b, float* out, int size);
void eltwise_sub(const float* a, const float* b, float* out, int size);
void eltwise_mul(const float* a, const float* b, float* out, int size);
void eltwise_div(const float* a, const float* b, float* out, int size);
void eltwise_pow(const float* a, const float* b, float* out, int size);

// Scalar operations
void scalar_add(const float* a, float val, float* out, int size);
void scalar_mul(const float* a, float val, float* out, int size);
void scalar_div(const float* a, float val, float* out, int size);
void scalar_pow(const float* a, float val, float* out, int size);

#define MAX_DIMS 4

struct TensorStrides {
    int data[MAX_DIMS];
};

// Broadcast wrappers
void eltwise_add_broadcast(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides);
void eltwise_sub_broadcast(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides);
void eltwise_mul_broadcast(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides);
void eltwise_div_broadcast(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides);
void eltwise_pow_broadcast(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides);

#endif