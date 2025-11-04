#include "tensor.h"
#include "func.h"

int main(){

    // Test1 Tensor类的实现
    std::cout << "------------------" <<std::endl;
    std::cout << "Test1 Tensor类的实现" << std::endl;
    std::cout << "初始化Tensor类" << std::endl;
    Tensor<float> t({3,2}, Device::CPU);
    for (int i = 0; i < t.get_size(); ++i){
        t.data()[i] = i - 3;
    }
    t.print();
    std::cout << std::endl;

    Tensor<float> g({2,3},Device::GPU);
    float* g_data = (float*) malloc(g.get_size());
    for (int i = 0; i < g.get_size(); ++i){
        g_data[i] = i*2 - 6;
    }
    cudaMemcpy(g.data(), g_data, g.get_size() * sizeof(float), cudaMemcpyHostToDevice);
    g.print();
    std::cout << std::endl;

    free(g_data);

    std::cout << "gpu()和cpu()函数的实现" << std::endl;
    Tensor<float> f = t.gpu();
    f.print();
    std::cout << std::endl;
    
    Tensor<float> h = g.cpu();
    h.print();
    std::cout << std::endl;

    std::cout << "多维Tensor和int数据类型的Tensor" << std::endl;
    Tensor<int> k({2,3,3}, Device::GPU);
    Tensor<int> k_cpu = k.cpu();
    for (int i = 0 ;i < k_cpu.get_size();++i){
        k_cpu.data()[i] = i - 4;
    }
    k = k_cpu.gpu();
    k.print();


    // Test2 relu和sigmoid函数的实现
    std::cout << "------------------" <<std::endl;
    std::cout << "Test2 relu函数的实现" << std::endl;
    Tensor<float> g_relu({2,3}, Device::GPU);
    relu_gpu<<<1, g.get_size()>>>(g.data(), g_relu.data(), g.get_size());
    g_relu.print();
    std::cout << "Test2 sigmoid函数的实现" << std::endl;
    Tensor<float> g_sigmoid({2,3}, Device::GPU);
    sigmoid_gpu<<<1, g.get_size()>>>(g.data(), g_sigmoid.data(), g.get_size());
    g_sigmoid.print();
    std::cout << std::endl;

    // Test3 relu_backward函数的实现
    std::cout << "----------------------------" <<std::endl;
    std::cout << "Test3 relu_backward函数的实现" << std::endl;
    Tensor<float> out_grad({2,3}, Device::GPU);
    Tensor<float> cpu_tmp = out_grad.cpu();
    for (int i = 0; i < out_grad.get_size(); ++i){
        cpu_tmp.data()[i] = i*2 - 3;  // -3, -1, 1, 3, 5, 7
    }
    out_grad = cpu_tmp.gpu();

    Tensor<float> in_grad({2,3}, Device::GPU);

    Tensor<float> x({2,3}, Device::GPU);
    Tensor<float> x_cpu_tmp = x.cpu();
    for (int i = 0; i < x.get_size(); ++i){
        x_cpu_tmp.data()[i] = i - 3; // -3, -2, -1, 0, 1, 2
    }
    x = x_cpu_tmp.gpu();

    relu_gpu_backward<<<1, x.get_size()>>>(in_grad.data(), out_grad.data(), x.data(), x.get_size());
    in_grad.print();

    std::cout << "Test3 sigmoid_backward函数的实现" << std::endl;
    sigmoid_gpu_backward<<<1, x.get_size()>>>(in_grad.data(), out_grad.data(), x.data(), x.get_size());
    in_grad.print();

}
