#include<stdio.h>
#include<cuda.h>

float relu_cpu(float x) {
    return x > 0 ? x : 0;
}

__global__ void relu_gpu(float* in, float* out){
    int i = threadIdx.x;
    out[i] = in[i] > 0 ? in[i] : 0;
}

void print(float* arr, int n){
    for (int i = 0; i < n; ++i){
        printf("%f ", arr[i]);
    }
    printf("\n");
}

int main(){
    const int N = 64;
    const int size = N * sizeof(float);
    float* h_in = (float*) malloc(size);
    float* h_out = (float*) malloc(size);
    for (int i = 0; i < N; ++i){
        h_in[i] = (i-32)*0.1;
    }
    for (int i = 0;i<N;++i){
        h_out[i] = relu_cpu(h_in[i]);
    }
    print(h_out, N);

    float* d_in = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    relu_gpu<<<1, N>>>(d_in, d_out);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    print(h_out, N);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}