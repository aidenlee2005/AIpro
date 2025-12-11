#include "layers.h"
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

__global__ void relu_gpu_kernel(float* in, float* out, int size){
    CUDA_KERNEL_LOOP(i, size){
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

__global__ void relu_gpu_backward_kernel(float* in_grad, float* out_grad, float* x, int size){
    CUDA_KERNEL_LOOP(i, size){
        in_grad[i] = x[i] > 0 ? out_grad[i] : 0;
    }
}

__global__ void sigmoid_gpu_kernel(float* in, float* out, int size){
    CUDA_KERNEL_LOOP(i, size){
        out[i] = 1 / (1 + exp(-in[i]));
    }
}

__global__ void sigmoid_gpu_backward_kernel(float* in_grad, float* out_grad, float* out, int size){
    CUDA_KERNEL_LOOP(i, size){
        float sig = out[i];
        in_grad[i] = sig * (1 - sig) * out_grad[i];
    }
}

void sigmoid_gpu(float* in, float* out, int size){
    sigmoid_gpu_kernel<<<((size + 255) / 256), 256, 0, 0>>>(in, out, size);
}

void sigmoid_gpu_backward(float* in_grad, float* out_grad, float* out, int size){
    sigmoid_gpu_backward_kernel<<<((size + 255) / 256), 256, 0, 0>>>(in_grad, out_grad, out, size);
}

void relu_gpu(float* in, float* out, int size){
    relu_gpu_kernel<<<((size + 255) / 256), 256, 0, 0>>>(in, out, size);
}

void relu_gpu_backward(float* in_grad, float* out_grad, float* x, int size){
    relu_gpu_backward_kernel<<<((size + 255) / 256), 256, 0, 0>>>(in_grad, out_grad, x, size);
}

__global__ void fill_elements(float* data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = value;
}

void gemm_gpu(TransposeType transA, TransposeType transB,
              const float* A, const float* B, float* C,
              const int m, const int n, const int k,
              const float alf, const float bet, cudaStream_t stream = 0){

    // 语义（row-major）: C (m x n) = alf * op(A) (m x k) * op(B) (k x n) + bet * C (m x n)
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS){
            std::cerr << "cublasCreate failed: " << stat << std::endl;
            return;
        }
    }

    cublasStatus_t stat = cublasSetStream(handle, stream);
    if (stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasSetStream failed: " << stat << std::endl;
        // Don't destroy handle here as it is static
        return;
    }

    const float* alpha = &alf;
    const float* beta  = &bet;

    // 将 row-major 映射到 cuBLAS(column-major)：计算 C_col = alpha * op_row(B)^T * op_row(A)^T + beta * C_col
    // 因此在 cublas 中传入 (B pointer) as A_arg, (A pointer) as B_arg，尺寸为 (n x m x k)
    cublasOperation_t opA_col = (transB == TransposeType::NoTranspose) ? CUBLAS_OP_N : CUBLAS_OP_T; // applied to B pointer (A_arg)
    cublasOperation_t opB_col = (transA == TransposeType::NoTranspose) ? CUBLAS_OP_N : CUBLAS_OP_T; // applied to A pointer (B_arg)

    // Leading dims: 当以 column-major 视角传入时，
    // - A_arg = B pointer 被视作列主序矩阵，其行数 = (transB == No) ? n : k
    // - B_arg = A pointer 被视作列主序矩阵，其行数 = (transA == No) ? k : m
    int lda = (transB == TransposeType::NoTranspose) ? n : k; // leading dim for B pointer (passed as A_arg)
    int ldb = (transA == TransposeType::NoTranspose) ? k : m; // leading dim for A pointer (passed as B_arg)
    int ldc = n; // C_col has rows = n

    stat = cublasSgemm(handle,
                       opA_col, opB_col,
                       /*m*/ n, /*n*/ m, /*k*/ k,
                       alpha,
                       B, lda,   // pass B as first operand to cublas
                       A, ldb,   // pass A as second operand
                       beta,
                       C, ldc);  // C is stored row-major, viewed as (n x m) in column-major
    if (stat != CUBLAS_STATUS_SUCCESS){
        std::cerr << "cublasSgemm failed: " << stat << std::endl;
    }

    // cublasDestroy(handle); // Do not destroy static handle
}

__global__ void quantize_one_decimal(float* data, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        data[idx] = roundf(data[idx] * 10.0f) / 10.0f;
    }
}

//random filling
void matrix_init_float(float* A, int n, unsigned long long seed = 123456){
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, A, n);
    curandDestroyGenerator(gen);

    int bs = 256;
    int gs = (n + bs - 1) / bs;
    quantize_one_decimal<<<gs, bs>>>(A, n);
}

__global__ void float_to_int_range(const float* src, int* dst, int n, int min_val, int range){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n){
        // 映射到 [min_val, min_val+range)
        int value = (int)(src[idx] * range) + min_val;
        dst[idx] = value;
    }
}

void matrix_init_int(int* A, int n, int min_val, int max_val, unsigned long long seed = 123456){
    float* tmp;
    cudaMalloc(&tmp, n * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, tmp, n);
    curandDestroyGenerator(gen);

    int range = max_val - min_val;
    int bs = 256;
    int gs = (n + bs - 1) / bs;
    float_to_int_range<<<gs, bs>>>(tmp, A, n, min_val, range);
    cudaFree(tmp);
}

//Task1: Fully Connected Layer
void forward_fc(float* input, float* output, float* weight, float* bias,
    int batch_size, int out_features, int in_features){
        //output(b, o) = input(b, i) * weight(i, o)
        gemm_gpu(TransposeType::NoTranspose,TransposeType::NoTranspose,
            input, weight, output, 
            batch_size, out_features, in_features, 
            1.0f, 0.0f);
        //output(b, o) += ones (b, 1) * bias(1, o)
        float* d_ones;
        cudaMalloc(&d_ones, batch_size * sizeof(float));
        cudaMemset(d_ones, 0, batch_size * sizeof(float));
        fill_elements<<<(batch_size + 255)/256, 256>>>(d_ones, batch_size, 1.0f);
        gemm_gpu(TransposeType::NoTranspose, TransposeType::NoTranspose,
            d_ones, bias, output, 
            batch_size, out_features, 1, 
            1.0f, 1.0f);
        cudaFree(d_ones);
    }

void backward_fc(float* input, float* weight, float* bias,
    int batch_size, int out_features, int in_features,
    float* grad_input, float* grad_output, float* grad_weight, float* grad_bias){
        // grad_input(b, i) = grad_output(b, o) * weight(i, o)^T
        gemm_gpu(TransposeType::NoTranspose, TransposeType::Transpose,
            grad_output, weight, grad_input, 
            batch_size, in_features, out_features, 
            1.0f, 0.0f);    
        // grad_weight(i, o) = input(b, i)^T * grad_output(b, o)
        gemm_gpu(TransposeType::Transpose, TransposeType::NoTranspose,
            input, grad_output, grad_weight, 
            in_features, out_features, batch_size, 
            1.0f, 0.0f);
        // grad_bias(1, o) = ones(1, b) * grad_output(b, o)
        float* d_ones;
        cudaMalloc(&d_ones, batch_size * sizeof(float));
        cudaMemset(d_ones, 0, batch_size * sizeof(float));
        fill_elements<<<(batch_size + 255)/256, 256>>>(d_ones, batch_size, 1.0f);
        gemm_gpu(TransposeType::Transpose, TransposeType::NoTranspose,
            d_ones, grad_output, grad_bias, 
            1, out_features, batch_size, 
            1.0f, 0.0f);
        cudaFree(d_ones);
    }
    

//Task2 : Convolutional Layer

__global__ void im2col_kernel(float* input_img, float* input_col,
            int batch_size, int in_channels, int height, int width){
    //Assume stride=1, padding=1, kernel_size=3
    // input_img(b, c, h, w) -> input_col()
    int col_h = height*width;
    int col_w = 3*3*in_channels;
    int col_row = blockIdx.x * blockDim.x + threadIdx.x;
    int col_col = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;
    if (col_row >= col_h || col_col >= col_w || batch >= batch_size) return;
    
    //计算输出位置
    int h_out = col_row / width;
    int w_out = col_row % width;
    //计算输入位置
    int channel = col_col / 9;
    int kernal_idx = col_col % 9;
    int kh = kernal_idx / 3;
    int kw = kernal_idx % 3;
    int h_in = h_out + kh - 1;
    int w_in = w_out + kw - 1;
    float value = 0.0f;
    if (h_in >=0 && h_in < height && w_in >=0 && w_in < width){
        int input_idx = batch * in_channels * height * width +
                        channel * height * width +
                        h_in * width +
                        w_in;
        value = input_img[input_idx];
    }
    int output_idx = batch * col_h * col_w +
                     col_row * col_w +
                     col_col;
    input_col[output_idx] = value;
}

void im2col(float* input_img, float* input_col,
            int batch_size, int in_channels, int height, int width,
            cudaStream_t stream){
    //Assume stride=1, padding=1, kernel_size=3
    int col_h = height * width;
    int col_w = 3 * 3 * in_channels;
    dim3 block(16, 16);
    dim3 grid((col_h + block.x -1)/block.x,
              (col_w + block.y -1)/block.y,
               batch_size);
    im2col_kernel<<<grid, block,0,stream>>>(input_img, input_col,
        batch_size, in_channels, height, width);
}

// BHWC -> BCHW
__global__ void nhwc_to_nchw_kernal(const float* bhwc, float* bchw,
                                         int batch, int out_channels, int height, int width){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * height * width;
    if (idx >= total) return;
    int oc = idx % out_channels;
    int r = idx / out_channels;
    int col_h = height * width;
    int b = r/col_h;
    int col_row = r % col_h;
    int h = col_row / width;
    int w = col_row % width;
    int out_idx = (b * out_channels * height * width) +
                  (oc * height * width) +
                  (h * width) +
                  w;
    bchw[out_idx] = bhwc[idx];
}

inline void nhwc_to_nchw(const float* bhwc, float* bchw,
                           int batch, int out_channels, int height, int width,
                           cudaStream_t stream){
    int total = batch * out_channels * height * width;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    nhwc_to_nchw_kernal<<<gridSize, blockSize, 0, stream>>>(bhwc, bchw,
                                                                 batch, out_channels, height, width);
    // 可选调试检查：
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) std::cerr << "outcol_to_nchw kernel launch failed: " << cudaGetErrorString(err) << std::endl;
}

// BCHW -> BHWC 
__global__ void nchw_to_nhwc_kernel(const float* bchw, float* bhwc,
                                      int batch, int out_channels, int height, int width){
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * out_channels * height * width;
    if (idx >= total) return;

    // 依据 NCHW 线性布局恢复坐标： idx = (((b * C + oc) * H) + h) * W + w
    int w = idx % width;
    size_t t = idx / width;
    int h = t % height;
    t = t / height;
    int oc = t % out_channels;
    int b = t / out_channels;

    int col_h = height * width;
    int outcol_idx = (b * col_h + h * width + w) * out_channels + oc;
    bhwc[outcol_idx] = bchw[idx];
}

inline void nchw_to_nhwc(const float* nchw, float* nhwc,
                           int batch, int out_channels, int height, int width,
                           cudaStream_t stream){
    if (batch <= 0 || out_channels <= 0 || height <= 0 || width <= 0) return;
    size_t total = (size_t)batch * out_channels * height * width;
    int blockSize = 256;
    size_t gridSize64 = (total + blockSize - 1) / blockSize;
    int gridSize = (int)std::min(gridSize64, (size_t)INT_MAX);
    nchw_to_nhwc_kernel<<<gridSize, blockSize, 0, stream>>>(nchw, nhwc,
                                                              batch, out_channels, height, width);
    // 可选调试：
    // cudaStreamSynchronize(stream);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) std::cerr << "nchw_to_nhwc kernel failed: " << cudaGetErrorString(err) << std::endl;
}

__global__ void col2im_kernal( float* grad_col, float* grad_input, 
           int batch_size, int in_channels, int height, int width){
    int col_h = height * width;
    int col_w = 3 * 3 * in_channels;
    int col_row = blockIdx.x * blockDim.x + threadIdx.x;
    int col_col = blockIdx.y * blockDim.y + threadIdx.y;
    int batch = blockIdx.z;
    if (col_row >= col_h || col_col >= col_w || batch >= batch_size) return;

    //计算输出位置
    int h_out = col_row / width;
    int w_out = col_row % width;
    //计算输入位置
    int channel = col_col / 9;
    int kernal_idx = col_col % 9;
    int kh = kernal_idx / 3;
    int kw = kernal_idx % 3;
    int h_in = h_out + kh - 1;
    int w_in = w_out + kw - 1;
    if (h_in >=0 && h_in < height && w_in >=0 && w_in < width){
        int input_idx = batch * in_channels * height * width +
                        channel * height * width +
                        h_in * width +
                        w_in;
        int col_idx = batch * col_h * col_w +
                      col_row * col_w +
                      col_col;
        float grad_val = grad_col[col_idx];
        atomicAdd(&grad_input[input_idx], grad_val);
    }
}

void col2im(float* grad_col, float* grad_input, 
           int batch_size, int in_channels, int height, int width,cudaStream_t stream){
    size_t im_size = (size_t)batch_size * in_channels * height * width;
    cudaError_t err = cudaMemsetAsync(grad_input, 0, im_size*sizeof(float), stream);
    if (err != cudaSuccess){
        std::cerr << "col2im cudaMemsetAsync failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    int col_h = height * width;
    int col_w = 3 * 3 * in_channels;
    dim3 block(16, 16);
    dim3 grid((col_h + block.x - 1)/block.x,
              (col_w + block.y - 1)/block.y,
               batch_size);
    col2im_kernal<<<grid, block,0,stream>>>(grad_col, grad_input,
        batch_size, in_channels, height, width);
}

void forward_conv2d(float* input, float* output, float* filter,
                int batch_size, int out_channels, int in_channels, int height, int width,
                cudaStream_t stream){
    //Assume stride=1, padding=1, kernel_size=3
    int col_h = height * width;
    int col_w = 3 * 3 * in_channels;
    float* d_input_col;
    cudaMalloc(&d_input_col, batch_size * col_h * col_w * sizeof(float));
    im2col(input, d_input_col, batch_size, in_channels, height, width, stream);

    float* d_output_col;
    cudaMalloc(&d_output_col, batch_size * col_h * out_channels * sizeof(float));

    //outcol(batch_size, col_h, out_channels) = input_col(batchsize, col_h, col_w) * filter(out_channels, in_channels, 3, 3) ^T
    //outcol((batch_size*col_h), out_channels) = input_col((batchsize*col_h), col_w) * filter(out_channels, (in_channels*3*3)) ^T
    gemm_gpu(TransposeType::NoTranspose, TransposeType::Transpose,
        d_input_col, filter, d_output_col,
        batch_size * col_h, out_channels, col_w,
        1.0f, 0.0f, stream);
    //move outcol(batch_size, height, weight out_channels) to output(batch_size, out_channels, height, width)
    nhwc_to_nchw(d_output_col, output, batch_size, out_channels, height, width, stream);
    cudaFree(d_output_col);
    cudaFree(d_input_col);
}

void backward_conv2d(float* input, float* filter,
        int batch_size, int out_channels, int in_channels, int height, int width,
        float* grad_input, float* grad_output, float* grad_filter,
        cudaStream_t stream){
    // Assume stride=1, padding=1, kernel_size=3
    int col_h = height * width;
    int col_w = 3 * 3 * in_channels;

    // 1) im2col(input) -> d_input_col
    float* d_input_col = nullptr;
    cudaMalloc(&d_input_col, (size_t)batch_size * col_h * col_w * sizeof(float));
    im2col(input, d_input_col, batch_size, in_channels, height, width, stream);

    // 2) convert grad_output (NCHW) -> outcol (batch*col_h, out_channels)
    float* d_grad_outcol = nullptr;
    cudaMalloc(&d_grad_outcol, (size_t)batch_size * col_h * out_channels * sizeof(float));
    nchw_to_nhwc(grad_output, d_grad_outcol, batch_size, out_channels, height, width, stream);

// // 调试片段：在调用 nchw_to_nhwc(...) 后插入（仅用于调试）
// cudaStreamSynchronize(stream); // 确保转换完成

// auto read_dev_one = [](const float* dptr, size_t idx){
//     float v=0.0f;
//     cudaMemcpy(&v, dptr + idx, sizeof(float), cudaMemcpyDeviceToHost);
//     return v;
// };

// int B = batch_size;
// int C = out_channels;
// int H = height;
// int W = width;
// int col_h_ = H * W;

// // 检查若干随机或固定位置
// std::vector<std::tuple<int,int,int,int>> checks = {
//     {0, 0, 0, 0},
//     {0, 1, 0, 1},
//     {0, C-1, H-1, W-1},
//     {B-1, C-1, H-1, W-1}
// };
// for (auto &t : checks){
//     int b,oc,h,w;
//     std::tie(b,oc,h,w) = t;
//     size_t nchw_idx   = ((size_t)b * C + oc) * H * W + (size_t)h * W + w;
//     size_t outcol_idx = ((size_t)b * col_h_ + (size_t)h * W + w) * C + oc;
//     float v_nchw = read_dev_one(grad_output, nchw_idx);
//     float v_out  = read_dev_one(d_grad_outcol, outcol_idx);
//     printf("check (b=%d,oc=%d,h=%d,w=%d) nchw[%zu]=%f outcol[%zu]=%f\n",
//            b,oc,h,w, nchw_idx, v_nchw, outcol_idx, v_out);
// }

    // 3) dW = d_grad_outcol^T * input_col
    // shapes: d_grad_outcol (batch*col_h, out_channels), d_input_col (batch*col_h, col_w)
    gemm_gpu(TransposeType::Transpose, TransposeType::NoTranspose,
        d_grad_outcol, d_input_col, grad_filter,
        out_channels, col_w, batch_size * col_h,
        1.0f, 0.0f, stream);

    // 4) dInput: grad_input_col = d_grad_outcol * filter
    float* d_grad_input_col = nullptr;
    cudaMalloc(&d_grad_input_col, (size_t)batch_size * col_h * col_w * sizeof(float));

    gemm_gpu(TransposeType::NoTranspose, TransposeType::NoTranspose,
        d_grad_outcol, filter, d_grad_input_col,
        batch_size * col_h, col_w, out_channels,
        1.0f, 0.0f, stream);
    // ensure GEMM finished before using d_grad_input_col on the same stream
    cudaStreamSynchronize(stream);

    // 5) col2im: accumulate grad_input_col -> grad_input (NCHW)
    col2im(d_grad_input_col, grad_input, batch_size, in_channels, height, width, stream);

    // 6) free temporaries
    cudaFree(d_grad_input_col);
    cudaFree(d_input_col);
    cudaFree(d_grad_outcol);
}

//Task 3: Max Pooling Layer

__global__ void max_pool_forward_kernel(const float* input, float* output, float* mask,
    int batch_size, int in_channels, int in_h, int in_w, int out_h, int out_w){
    //Assume kernel_size=2, stride=2
    int nthreads = batch_size * in_channels * out_h * out_w;
    CUDA_KERNEL_LOOP(idx, nthreads){
        int b = idx / (in_channels * out_h * out_w);  //batch index
        int c = (idx % (in_channels * out_h * out_w)) / (out_h * out_w); //channel index
        int h = (idx % (out_h * out_w)) / out_w; //height index
        int w = idx % out_w; //width index
        int h_in = h * 2;
        int w_in = w * 2;
        int max_idx = -1;
        float max_val = -INT32_MAX;
        for (int kh = 0; kh < 2; kh++){
            for (int kw = 0; kw < 2; kw++){
                int h_idx = h_in + kh;
                int w_idx = w_in + kw;
                if (h_idx >= 0 && h_idx < in_h && w_idx >= 0 && w_idx < in_w){
                    int input_idx = b * in_channels * in_h * in_w + c * in_h * in_w + h_idx * in_w + w_idx; 
                    float val = input[input_idx];
                    if (val > max_val){
                        max_val = val;
                        max_idx = input_idx;
                    }
                }
            }
        }
        output[idx] = max_val;
        mask[idx] = (float)max_idx;
    }
}

void forward_maxpool(const float* input, float* output, float* mask,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_h, int out_w, cudaStream_t stream){
        //Assume kernel_size=2, stride=2
        int nthreads = batch_size * in_channels * out_h * out_w;
        int bs = 256;
        int gs = (nthreads + bs - 1) / bs;
        max_pool_forward_kernel<<<gs, bs, 0, stream>>>(input, output, mask,
            batch_size, in_channels, in_h, in_w, out_h, out_w);
    }

__global__ void max_pool_backward_kernel(const float* grad_output, const float* mask, float* grad_input,
    int batch_size, int in_channels, int in_h, int in_w, int out_h, int out_w){
        int nthreads = batch_size * in_channels * out_h * out_w;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nthreads) return;
        int in_idx = (int)mask[idx];
        size_t in_size = (size_t)batch_size * in_channels * in_h * in_w;
        if (in_idx >= 0 && in_idx < in_size){
            atomicAdd(&grad_input[in_idx], grad_output[idx]);
        }
    }

void backward_maxpool(const float* grad_output, const float* mask, float* grad_input,
    int batch_size, int in_channels, int in_h, int in_w,
    int out_h, int out_w, cudaStream_t stream){
        //Assume kernel_size=2, stride=2
        int nthreads = batch_size * in_channels * out_h * out_w;
        int bs = 256;
        int gs = (nthreads + bs - 1) / bs;
        //initialize grad_input to zero
        size_t im_size = (size_t)batch_size * in_channels * in_h * in_w;
        cudaError_t err = cudaMemsetAsync(grad_input, 0, im_size*sizeof(float), stream);
        //scatter grad_output to grad_input according to mask
        max_pool_backward_kernel<<<gs, bs, 0, stream>>>(grad_output, mask, grad_input,
            batch_size, in_channels, in_h, in_w, out_h, out_w);
    }

//Task4: Softmax Layer
__global__ void softmax_forward_kernel(const float* input, float* output,
    int batch_size, int num_classes){
    int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float buf[]; //用于reduce
    const float* input_row = input + row * num_classes;
    float* output_row = output + row * num_classes;

    float local_max = -INT32_MAX;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x){
        if (input_row[i] > local_max) local_max = input_row[i];
    }
    buf[threadIdx.x] = local_max;
    __syncthreads();
    //reduce max
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (threadIdx.x < s){
            if (buf[threadIdx.x + s] > buf[threadIdx.x]){
                buf[threadIdx.x] = buf[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    float row_max = buf[0];
    //compute exp 
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x){
        float val = expf(input_row[i] - row_max);
        output_row[i] = val;
        local_sum += val;
    }
    buf[threadIdx.x] = local_sum;
    __syncthreads();
    //reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (threadIdx.x < s){
            buf[threadIdx.x] += buf[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = buf[0];
    //compute softmax
    float inv_row_sum = (row_sum != 0.0f) ? (1.0f / row_sum) : 0.0f;;
    for (int i = threadIdx.x; i < num_classes; i += blockDim.x){
        output_row[i] *= inv_row_sum;
    }
}

void forward_softmax(const float* input, float* output,
    int batch_size ,int num_classes, cudaStream_t stream){
    //input shape (batch_size, num_classes)
    //output shape (batch_size, num_classes)
    int bs = 256;
    dim3 grid(batch_size);
    softmax_forward_kernel<<<grid, bs, /*shared memory*/bs * sizeof(float), stream>>>(input, output,
        batch_size, num_classes);
}

//Task5: Cross Entropy Loss Layer

__inline__ __device__ float blockReduceSum(float val) {
    extern __shared__ float buf[];
    int tid = threadIdx.x;
    buf[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) buf[tid] += buf[tid + s];
        __syncthreads();
    }
    return buf[0];
}

// ---------------- Cross Entropy 修正 ----------------

// loss_kernel: 采用 one-block-per-sample 的简单实现，避免不匹配的 block/thread 归约
__global__ void loss_kernel(const float* input, const float* labels, float* loss,
    int batch_size, int num_classes){
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // 只由线程0读取标签并计算该行的负对数似然，再原子加到全局 loss
    if (threadIdx.x == 0) {
        int label = int(labels[row]);
        float prob = input[row * num_classes + label];
        prob = fmaxf(prob, 1e-12f);  // 防止 log(0)
        float local_loss = -logf(prob);
        atomicAdd(loss, local_loss);
    }
}

// forward_cross_entropy: 先复用 forward_softmax 写入临时 d_softmax，再调用 loss_kernel
void forward_cross_entropy(const float* input, const float* labels, float* loss,
    int batch_size, int num_classes, cudaStream_t stream){
    // input: logits (batch_size, num_classes)
    // labels: (batch_size)
    // loss: pointer to single float (output)

    float* d_softmax = nullptr;
    size_t total = (size_t)batch_size * num_classes;
    cudaMalloc(&d_softmax, total * sizeof(float));

    // logits -> softmax
    forward_softmax(input, d_softmax, batch_size, num_classes, stream);

    float h_zero = 0.0f;
    float* d_loss = nullptr;
    cudaMalloc(&d_loss, sizeof(float));
    cudaMemcpyAsync(d_loss, &h_zero, sizeof(float), cudaMemcpyHostToDevice, stream);

    // launch one block per sample (softmax_forward already uses this convention)
    int threads = 256;
    int blocks = batch_size;
    size_t shared_mem = 0; // not used in this loss_kernel implementation

    loss_kernel<<<blocks, threads, shared_mem, stream>>>(d_softmax, labels, d_loss,
        batch_size, num_classes);

    float h_loss = 0.0f;
    cudaMemcpyAsync(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    *loss = h_loss / batch_size;

    cudaFree(d_loss);
    cudaFree(d_softmax);
}

// subtract_labels: 每行一个 block，线程在列上循环，安全处理任意 num_classes
__global__ void subtract_labels(float* grad_input, const float* labels, int batch_size, int num_classes){
    int row = blockIdx.x;
    if (row >= batch_size) return;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int label = int(labels[row]);
    // 每个线程处理多个 class 列
    for (int c = tid; c < num_classes; c += stride){
        if (c == label){
            grad_input[row * num_classes + c] -= 1.0f;
        }
    }
}

// scale_grad: 通过全局索引循环覆盖所有元素
__global__ void scale_grad(float* grad_input, int n, float scale){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride){
        grad_input[i] *= scale;
    }
}

// backward_cross_entropy: 先写 softmax 到 grad_output，再做 subtract_labels 与 scale
void backward_cross_entropy(const float* input_logits, const float* labels,
    int batch_size, int num_classes, float* grad_output, cudaStream_t stream){
    // write softmax(logits) into grad_output
    forward_softmax(input_logits, grad_output, batch_size, num_classes, stream);

    // grad_output(b, c) -= 1 if c == labels[b]
    int threads_per_block = (num_classes < 256) ? num_classes : 256;
    if (threads_per_block < 32) threads_per_block = 32; // 保证至少一个warp
    int blocks_labels = batch_size;
    subtract_labels<<<blocks_labels, threads_per_block, 0, stream>>>(grad_output, labels, batch_size, num_classes);

    // grad_output /= batch_size (对所有元素)
    int total = batch_size * num_classes;
    int bs = 256;
    int gs = (total + bs - 1) / bs;
    scale_grad<<<gs, bs, 0, stream>>>(grad_output, total, 1.0f / float(batch_size));
}

// SGD Update Kernel
__global__ void sgd_update_kernel(float* param, const float* grad, float* velocity,
                                  float lr, float momentum, float weight_decay, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx];
        if (weight_decay != 0.0f) {
            g += param[idx] * weight_decay;
        }
        
        float v = 0.0f;
        if (momentum != 0.0f) {
            // velocity is assumed to be initialized to 0
            v = velocity[idx] * momentum + g;
            velocity[idx] = v;
            param[idx] -= lr * v;
        } else {
            param[idx] -= lr * g;
        }
    }
}

void sgd_update_gpu(float* param, const float* grad, float* velocity, 
                    float lr, float momentum, float weight_decay, int size, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    sgd_update_kernel<<<gridSize, blockSize, 0, stream>>>(param, grad, velocity, lr, momentum, weight_decay, size);
}

// Adam Update Kernel
__global__ void adam_update_kernel(float* param, const float* grad, float* m, float* v,
                                   float lr, float beta1, float beta2, float eps, float weight_decay, 
                                   int step, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx];
        if (weight_decay != 0.0f) {
            g += param[idx] * weight_decay;
        }
        
        // Update biased first moment estimate
        float m_t = m[idx] * beta1 + g * (1.0f - beta1);
        m[idx] = m_t;
        
        // Update biased second raw moment estimate
        float v_t = v[idx] * beta2 + (g * g) * (1.0f - beta2);
        v[idx] = v_t;
        
        // Compute bias-corrected first moment estimate
        float m_hat = m_t / (1.0f - powf(beta1, step));
        
        // Compute bias-corrected second raw moment estimate
        float v_hat = v_t / (1.0f - powf(beta2, step));
        
        // Update parameters
        param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

void adam_update_gpu(float* param, const float* grad, float* m, float* v,
                     float lr, float beta1, float beta2, float eps, float weight_decay, 
                     int step, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    adam_update_kernel<<<blocks, threads, 0, stream>>>(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step, size);
}

// Element-wise kernels
__global__ void eltwise_add_kernel(const float* a, const float* b, float* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = a[i] + b[i];
    }
}

__global__ void eltwise_sub_kernel(const float* a, const float* b, float* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = a[i] - b[i];
    }
}

__global__ void eltwise_mul_kernel(const float* a, const float* b, float* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = a[i] * b[i];
    }
}

__global__ void eltwise_div_kernel(const float* a, const float* b, float* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = a[i] / b[i];
    }
}

__global__ void eltwise_pow_kernel(const float* a, const float* b, float* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = powf(a[i], b[i]);
    }
}

// Scalar kernels
__global__ void scalar_add_kernel(const float* a, float val, float* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = a[i] + val;
    }
}

__global__ void scalar_mul_kernel(const float* a, float val, float* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = a[i] * val;
    }
}

__global__ void scalar_div_kernel(const float* a, float val, float* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = a[i] / val;
    }
}

__global__ void scalar_pow_kernel(const float* a, float val, float* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = powf(a[i], val);
    }
}

// Wrappers
void eltwise_add(const float* a, const float* b, float* out, int size) {
    eltwise_add_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

void eltwise_sub(const float* a, const float* b, float* out, int size) {
    eltwise_sub_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

void eltwise_mul(const float* a, const float* b, float* out, int size) {
    eltwise_mul_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

void eltwise_div(const float* a, const float* b, float* out, int size) {
    eltwise_div_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

void eltwise_pow(const float* a, const float* b, float* out, int size) {
    eltwise_pow_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size);
}

void scalar_add(const float* a, float val, float* out, int size) {
    scalar_add_kernel<<<(size + 255) / 256, 256>>>(a, val, out, size);
}

void scalar_mul(const float* a, float val, float* out, int size) {
    scalar_mul_kernel<<<(size + 255) / 256, 256>>>(a, val, out, size);
}

void scalar_div(const float* a, float val, float* out, int size) {
    scalar_div_kernel<<<(size + 255) / 256, 256>>>(a, val, out, size);
}

void scalar_pow(const float* a, float val, float* out, int size) {
    scalar_pow_kernel<<<(size + 255) / 256, 256>>>(a, val, out, size);
}

// Broadcast Kernels
__global__ void eltwise_add_broadcast_kernel(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int temp_idx = idx;
    int idx_a = 0;
    int idx_b = 0;

    for (int i = 0; i < ndim; ++i) {
        int coord = temp_idx / out_strides.data[i];
        temp_idx %= out_strides.data[i];
        idx_a += coord * a_strides.data[i];
        idx_b += coord * b_strides.data[i];
    }
    out[idx] = a[idx_a] + b[idx_b];
}

__global__ void eltwise_sub_broadcast_kernel(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int temp_idx = idx;
    int idx_a = 0;
    int idx_b = 0;

    for (int i = 0; i < ndim; ++i) {
        int coord = temp_idx / out_strides.data[i];
        temp_idx %= out_strides.data[i];
        idx_a += coord * a_strides.data[i];
        idx_b += coord * b_strides.data[i];
    }
    out[idx] = a[idx_a] - b[idx_b];
}

__global__ void eltwise_mul_broadcast_kernel(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int temp_idx = idx;
    int idx_a = 0;
    int idx_b = 0;

    for (int i = 0; i < ndim; ++i) {
        int coord = temp_idx / out_strides.data[i];
        temp_idx %= out_strides.data[i];
        idx_a += coord * a_strides.data[i];
        idx_b += coord * b_strides.data[i];
    }
    out[idx] = a[idx_a] * b[idx_b];
}

__global__ void eltwise_div_broadcast_kernel(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int temp_idx = idx;
    int idx_a = 0;
    int idx_b = 0;

    for (int i = 0; i < ndim; ++i) {
        int coord = temp_idx / out_strides.data[i];
        temp_idx %= out_strides.data[i];
        idx_a += coord * a_strides.data[i];
        idx_b += coord * b_strides.data[i];
    }
    out[idx] = a[idx_a] / b[idx_b];
}

__global__ void eltwise_pow_broadcast_kernel(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    int temp_idx = idx;
    int idx_a = 0;
    int idx_b = 0;

    for (int i = 0; i < ndim; ++i) {
        int coord = temp_idx / out_strides.data[i];
        temp_idx %= out_strides.data[i];
        idx_a += coord * a_strides.data[i];
        idx_b += coord * b_strides.data[i];
    }
    out[idx] = powf(a[idx_a], b[idx_b]);
}

// Broadcast Wrappers
void eltwise_add_broadcast(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides) {
    eltwise_add_broadcast_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size, ndim, out_strides, a_strides, b_strides);
}

void eltwise_sub_broadcast(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides) {
    eltwise_sub_broadcast_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size, ndim, out_strides, a_strides, b_strides);
}

void eltwise_mul_broadcast(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides) {
    eltwise_mul_broadcast_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size, ndim, out_strides, a_strides, b_strides);
}

void eltwise_div_broadcast(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides) {
    eltwise_div_broadcast_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size, ndim, out_strides, a_strides, b_strides);
}

void eltwise_pow_broadcast(const float* a, const float* b, float* out, int size, int ndim, TensorStrides out_strides, TensorStrides a_strides, TensorStrides b_strides) {
    eltwise_pow_broadcast_kernel<<<(size + 255) / 256, 256>>>(a, b, out, size, ndim, out_strides, a_strides, b_strides);
}

// --- BatchNorm Implementation ---

__global__ void batch_norm_collect_statistics_kernel(
    const float* input, float* mean, float* var,
    int batch_size, int channels, int height, int width) {
    
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int spatial_size = height * width;
    int num_elements = batch_size * spatial_size;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / spatial_size;
        int hw = i % spatial_size;
        int idx = n * channels * spatial_size + c * spatial_size + hw;
        float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }
    
    __shared__ float s_sum[256]; //shared memory
    __shared__ float s_sum_sq[256];
    
    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float m = s_sum[0] / num_elements;
        mean[c] = m;
        var[c] = s_sum_sq[0] / num_elements - m * m;
    }
}

__global__ void batch_norm_forward_kernel(
    const float* input, float* output,
    const float* mean, const float* var,
    const float* weight, const float* bias,
    int batch_size, int channels, int height, int width,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = height * width;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        
        float m = mean[c];
        float v = var[c];
        float inv_std = rsqrtf(v + eps);
        
        float val = input[idx];
        float norm = (val - m) * inv_std;
        
        float w = (weight) ? weight[c] : 1.0f;
        float b = (bias) ? bias[c] : 0.0f;
        
        output[idx] = norm * w + b;
    }
}

__global__ void batch_norm_forward_inference_kernel(
    const float* input, float* output,
    const float* running_mean, const float* running_var,
    const float* weight, const float* bias,
    int batch_size, int channels, int height, int width,
    float eps) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = height * width;
    int total_size = batch_size * channels * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        
        float m = running_mean[c];
        float v = running_var[c];
        float inv_std = rsqrtf(v + eps);
        
        float val = input[idx];
        float norm = (val - m) * inv_std;
        
        float w = (weight) ? weight[c] : 1.0f;
        float b = (bias) ? bias[c] : 0.0f;
        
        output[idx] = norm * w + b;
    }
}

__global__ void batch_norm_save_stats_kernel(
    const float* mean, const float* var,
    float* save_mean, float* save_inv_std,
    int channels, float eps) {
    
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < channels) {
        save_mean[c] = mean[c];
        save_inv_std[c] = rsqrtf(var[c] + eps);
    }
}

__global__ void update_running_stats_kernel(
    float* running_mean, float* running_var,
    const float* current_mean, const float* current_var,
    float momentum, int channels) {
    
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < channels) {
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * current_mean[c];
        running_var[c] = (1.0f - momentum) * running_var[c] + momentum * current_var[c];
    }
}

void batch_norm_forward_training(
    float* input, float* output, 
    float* weight, float* bias,
    float* running_mean, float* running_var,
    float* save_mean, float* save_inv_std,
    int batch_size, int channels, int height, int width,
    float momentum, float eps) {
    
    thrust::device_vector<float> temp_mean(channels);
    thrust::device_vector<float> temp_var(channels);
    
    float* d_mean = thrust::raw_pointer_cast(temp_mean.data());
    float* d_var = thrust::raw_pointer_cast(temp_var.data());
    
    batch_norm_collect_statistics_kernel<<<channels, 256>>>(
        input, d_mean, d_var, batch_size, channels, height, width);
        
    batch_norm_forward_kernel<<<(batch_size * channels * height * width + 255) / 256, 256>>>(
        input, output, d_mean, d_var, weight, bias, batch_size, channels, height, width, eps);
        
    batch_norm_save_stats_kernel<<<(channels + 255) / 256, 256>>>(
        d_mean, d_var, save_mean, save_inv_std, channels, eps);
        
    update_running_stats_kernel<<<(channels + 255) / 256, 256>>>(
        running_mean, running_var, d_mean, d_var, momentum, channels);
}

void batch_norm_forward_inference(
    float* input, float* output,
    float* weight, float* bias,
    float* running_mean, float* running_var,
    int batch_size, int channels, int height, int width,
    float eps) {
    
    batch_norm_forward_inference_kernel<<<(batch_size * channels * height * width + 255) / 256, 256>>>(
        input, output, running_mean, running_var, weight, bias, batch_size, channels, height, width, eps);
}

__global__ void batch_norm_backward_reduce_kernel(
    const float* grad_output, const float* input,
    const float* mean, const float* inv_std,
    float* grad_weight, float* grad_bias,
    int batch_size, int channels, int height, int width) {
    
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int spatial_size = height * width;
    int num_elements = batch_size * spatial_size;
    
    float sum_dy = 0.0f;
    float sum_dy_xhat = 0.0f;
    
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / spatial_size;
        int hw = i % spatial_size;
        int idx = n * channels * spatial_size + c * spatial_size + hw;
        
        float dy = grad_output[idx];
        float x = input[idx];
        float m = mean[c];
        float is = inv_std[c];
        float x_hat = (x - m) * is;
        
        sum_dy += dy;
        sum_dy_xhat += dy * x_hat;
    }
    
    __shared__ float s_sum_dy[256];
    __shared__ float s_sum_dy_xhat[256];
    
    s_sum_dy[tid] = sum_dy;
    s_sum_dy_xhat[tid] = sum_dy_xhat;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum_dy[tid] += s_sum_dy[tid + stride];
            s_sum_dy_xhat[tid] += s_sum_dy_xhat[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        if (grad_bias) grad_bias[c] = s_sum_dy[0];
        if (grad_weight) grad_weight[c] = s_sum_dy_xhat[0];
    }
}

__global__ void batch_norm_backward_input_kernel(
    const float* grad_output, const float* input, float* grad_input,
    const float* mean, const float* inv_std,
    const float* weight,
    const float* sum_dy, const float* sum_dy_xhat,
    int batch_size, int channels, int height, int width) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial_size = height * width;
    int total_size = batch_size * channels * spatial_size;
    int num_elements = batch_size * spatial_size;
    
    if (idx < total_size) {
        int c = (idx / spatial_size) % channels;
        
        float m = mean[c];
        float is = inv_std[c];
        float w = (weight) ? weight[c] : 1.0f;
        
        float dy = grad_output[idx];
        float x = input[idx];
        float x_hat = (x - m) * is;
        
        float s_dy = sum_dy[c];
        float s_dy_xhat = sum_dy_xhat[c];
        
        // dx = (gamma / std) * (dy - mean(dy) - x_hat * mean(dy * x_hat))
        // mean(dy) = sum_dy / N
        // mean(dy * x_hat) = sum_dy_xhat / N
        
        float term1 = dy;
        float term2 = s_dy / num_elements;
        float term3 = x_hat * (s_dy_xhat / num_elements);
        
        grad_input[idx] = w * is * (term1 - term2 - term3);
    }
}

void batch_norm_backward(
    float* grad_output, float* input, float* grad_input,
    float* weight, float* grad_weight, float* grad_bias,
    float* save_mean, float* save_inv_std,
    int batch_size, int channels, int height, int width) {
    
    // We need temporary storage for sum_dy and sum_dy_xhat if grad_weight/grad_bias are null
    // But usually they are provided.
    // However, the backward_input_kernel needs them.
    // So we should compute them into grad_weight/grad_bias (if provided) or temp.
    
    // Assuming grad_weight and grad_bias are provided and allocated.
    
    batch_norm_backward_reduce_kernel<<<channels, 256>>>(
        grad_output, input, save_mean, save_inv_std, grad_weight, grad_bias,
        batch_size, channels, height, width);
        
    batch_norm_backward_input_kernel<<<(batch_size * channels * height * width + 255) / 256, 256>>>(
        grad_output, input, grad_input, save_mean, save_inv_std, weight,
        grad_weight, grad_bias, batch_size, channels, height, width);
}
