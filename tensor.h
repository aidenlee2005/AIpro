#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <memory>
#include <cstring> // 添加以支持 memcpy

enum class Device{
    CPU,
    GPU
};

template<typename T>
class Tensor{
private:
    std::vector<int> shape;
    std::vector<int> strides;
    std::unique_ptr<T[]> h_data;
    T* d_data = nullptr;
    Device device;
    int size{1};

public:
    Tensor(std::vector<int> s, Device d):shape(s), device(d) {
        for (int dim: shape){
            size *= dim;
        }
        int k = size;
        for (int dim: shape){
            k = k/dim;
            strides.push_back(k);
        }
        if (device == Device::CPU){
            h_data = std::make_unique<T[]>(size);
        }
        else{
            cudaMalloc(&d_data, size * sizeof(T));
        }
    }

    Tensor(Tensor&& other) noexcept:  //移动构造函数
        shape(std::move(other.shape)),
        strides(std::move(other.strides)),
        h_data(std::move(other.h_data)),
        d_data(other.d_data),
        device(other.device),
        size(other.size){
            other.d_data = nullptr;
            other.size = 0;
        }

    Tensor& operator=(Tensor&& other) noexcept {  //移动赋值运算符
        if (this != &other) {
            if (device == Device::GPU && d_data != nullptr) {
                cudaFree(d_data);
            }
            shape = std::move(other.shape);
            strides = std::move(other.strides);
            h_data = std::move(other.h_data);
            d_data = other.d_data;
            device = other.device;
            size = other.size;

            other.d_data = nullptr; //释放内存
            other.size = 0;
        }
        return *this;
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    T* data(){
        if (device == Device::CPU){
            return h_data.get();
        }
        else{
            return d_data;
        }
    }

    std::vector<int> get_shape(){
        return shape;
    }

    int get_size(){
        return size;
    }

    ~Tensor(){
        if (device == Device::GPU){
            if (d_data != nullptr)
                cudaFree(d_data);
        }
        else if (device == Device::CPU){
            h_data.reset();
        }
    }

    Tensor cpu(){
        if (device == Device::CPU){
            return std::move(*this);
        }
        Tensor t(shape, Device::CPU);
        cudaMemcpy(t.data(), d_data, size * sizeof(T), cudaMemcpyDeviceToHost);
        return std::move(t);
    }

    Tensor gpu(){
        if (device == Device::GPU){
            return std::move(*this);
        }
        Tensor t(shape, Device::GPU);
        cudaMemcpy(t.data(), h_data.get(), size * sizeof(T), cudaMemcpyHostToDevice);
        return std::move(t);
    }

    bool is_cpu() const { return device == Device::CPU; }
    bool is_gpu() const { return device == Device::GPU; }

    void copy_to_host(T* out) const {
        if (device == Device::CPU) {
            std::memcpy(out, h_data.get(), size * sizeof(T));
        } else {
            cudaMemcpy(out, d_data, size * sizeof(T), cudaMemcpyDeviceToHost);
        }
    }

    void copy_from_host(const T* in) {
        if (device == Device::CPU) {
            std::memcpy(h_data.get(), in, size * sizeof(T));
        } else {
            cudaMemcpy(d_data, in, size * sizeof(T), cudaMemcpyHostToDevice);
        }
    }

    void print(){
        if (device == Device::CPU){
            int shapedim = shape.size();
            if (shapedim == 1){
                std::cout << "[ ";
                for (int i = 0; i < size; i++){
                    std::cout << h_data[i] << " ";
                }
                std::cout << "]" << std::endl;
                return;
            }
            if (shapedim == 2 && shape[1] == 1) {
                std::cout << "[";
                for (int i = 0; i < shape[0]; i++) {
                    if (i!= 0) std::cout << " ";
                    std::cout << "[ " << h_data[i] << " ]";
                    if (i!= shape[0]-1) std::cout << "\n";
                }
                std::cout << "]" << std::endl;
                return;
            }
            for (int i = 0; i < shapedim; i++)
                std::cout << '[';
            std::cout << ' ';
            for (int i = 0; i < size; i++){
                std::cout << h_data[i] << " ";
                int count = 0;
                for (int dim: strides){
                    if (dim!= 1 && (i+1) % dim == 0 && i!= size-1)
                        count ++;
                }
                for (int j = 0; j < count; j++)
                    std::cout << ']';
                if (count != 0){
                    std::cout << std::endl;
                    for (int j = 0; j < shapedim - count ;j++)
                        std::cout << ' ';
                }
                for (int j = 0; j < count; j++)
                    std::cout << '[';
                if (count != 0)
                    std::cout << ' ';                
            }
            for (int i = 0; i < shape.size(); i++)
                std::cout << ']';
            std::cout << std::endl;
        }
        else if (device == Device::GPU){
            Tensor t = this->cpu();
            t.print();
        }
    }

};


#endif //TENSOR_H
