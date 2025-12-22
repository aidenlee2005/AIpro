#include "layers.h"
#include "memory_pool.h"
#include <map>
#include <tuple>
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

// ===========FC=============

void forward_fc(float* input, float* output, float* weight, float* bias,
    int batch_size, int out_features, int in_features){
        //output(b, o) = input(b, i) * weight(i, o)
        gemm_gpu(TransposeType::NoTranspose,TransposeType::NoTranspose,
            input, weight, output, 
            batch_size, out_features, in_features, 
            1.0f, 0.0f);
        //output(b, o) += ones (b, 1) * bias(1, o)
        size_t ones_size = batch_size * sizeof(float);
        float* d_ones = (float*)MemoryPool::instance().allocate(ones_size);
        // cudaMemset(d_ones, 0, batch_size * sizeof(float)); // Removed redundant memset
        fill_elements<<<(batch_size + 255)/256, 256>>>(d_ones, batch_size, 1.0f);
        gemm_gpu(TransposeType::NoTranspose, TransposeType::NoTranspose,
            d_ones, bias, output, 
            batch_size, out_features, 1, 
            1.0f, 1.0f);
        MemoryPool::instance().deallocate(d_ones, ones_size);
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
        size_t ones_size = batch_size * sizeof(float);
        float* d_ones = (float*)MemoryPool::instance().allocate(ones_size);
        // cudaMemset(d_ones, 0, batch_size * sizeof(float)); // Removed redundant memset
        fill_elements<<<(batch_size + 255)/256, 256>>>(d_ones, batch_size, 1.0f);
        gemm_gpu(TransposeType::Transpose, TransposeType::NoTranspose,
            d_ones, grad_output, grad_bias, 
            1, out_features, batch_size, 
            1.0f, 0.0f);
        MemoryPool::instance().deallocate(d_ones, ones_size);
    }
    

// ===========Conv2d=============

__global__ void im2col_kernel(float* input_img, float* input_col,
            int batch_size, int in_channels, int height, int width){
    //Assume stride=1, padding=1, kernel_size=3
    // input_img(b, c, h, w) -> input_col()
    int col_h = height*width;
    int col_w = 3*3*in_channels;
    
    // Swap mapping: threadIdx.x maps to col_col (inner dimension) for coalesced write
    int col_col = blockIdx.x * blockDim.x + threadIdx.x;
    int col_row = blockIdx.y * blockDim.y + threadIdx.y;
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
    // Swap grid dimensions to match swapped kernel mapping
    dim3 grid((col_w + block.x -1)/block.x,
              (col_h + block.y -1)/block.y,
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
}


__global__ void col2im_kernal( float* grad_col, float* grad_input, 
           int batch_size, int in_channels, int height, int width){
    int col_h = height * width;
    int col_w = 3 * 3 * in_channels;
    
    // Swap mapping: threadIdx.x maps to col_col (inner dimension) for coalesced read
    int col_col = blockIdx.x * blockDim.x + threadIdx.x;
    int col_row = blockIdx.y * blockDim.y + threadIdx.y;
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
    // Swap grid dimensions to match swapped kernel mapping
    dim3 grid((col_w + block.x - 1)/block.x,
              (col_h + block.y - 1)/block.y,
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
    
    size_t input_col_size = (size_t)batch_size * col_h * col_w * sizeof(float);
    float* d_input_col = (float*)MemoryPool::instance().allocate(input_col_size);
    im2col(input, d_input_col, batch_size, in_channels, height, width, stream);

    size_t output_col_size = (size_t)batch_size * col_h * out_channels * sizeof(float);
    float* d_output_col = (float*)MemoryPool::instance().allocate(output_col_size);

    //outcol(batch_size, col_h, out_channels) = input_col(batchsize, col_h, col_w) * filter(out_channels, in_channels, 3, 3) ^T
    //outcol((batch_size*col_h), out_channels) = input_col((batchsize*col_h), col_w) * filter(out_channels, (in_channels*3*3)) ^T
    gemm_gpu(TransposeType::NoTranspose, TransposeType::Transpose,
        d_input_col, filter, d_output_col,
        batch_size * col_h, out_channels, col_w,
        1.0f, 0.0f, stream);
    //move outcol(batch_size, height, weight out_channels) to output(batch_size, out_channels, height, width)
    nhwc_to_nchw(d_output_col, output, batch_size, out_channels, height, width, stream);
    
    MemoryPool::instance().deallocate(d_output_col, output_col_size);
    MemoryPool::instance().deallocate(d_input_col, input_col_size);
}

void backward_conv2d(float* input, float* filter,
        int batch_size, int out_channels, int in_channels, int height, int width,
        float* grad_input, float* grad_output, float* grad_filter,
        cudaStream_t stream){
    // Assume stride=1, padding=1, kernel_size=3
    int col_h = height * width;
    int col_w = 3 * 3 * in_channels;

    // 1) im2col(input) -> d_input_col
    size_t input_col_size = (size_t)batch_size * col_h * col_w * sizeof(float);
    float* d_input_col = (float*)MemoryPool::instance().allocate(input_col_size);
    im2col(input, d_input_col, batch_size, in_channels, height, width, stream);

    // 2) convert grad_output (NCHW) -> outcol (batch*col_h, out_channels)
    size_t grad_outcol_size = (size_t)batch_size * col_h * out_channels * sizeof(float);
    float* d_grad_outcol = (float*)MemoryPool::instance().allocate(grad_outcol_size);
    nchw_to_nhwc(grad_output, d_grad_outcol, batch_size, out_channels, height, width, stream);

    // 3) dW = d_grad_outcol^T * input_col
    // shapes: d_grad_outcol (batch*col_h, out_channels), d_input_col (batch*col_h, col_w)
    gemm_gpu(TransposeType::Transpose, TransposeType::NoTranspose,
        d_grad_outcol, d_input_col, grad_filter,
        out_channels, col_w, batch_size * col_h,
        1.0f, 0.0f, stream);

    // 4) dInput: grad_input_col = d_grad_outcol * filter
    size_t grad_input_col_size = (size_t)batch_size * col_h * col_w * sizeof(float);
    float* d_grad_input_col = (float*)MemoryPool::instance().allocate(grad_input_col_size);

    gemm_gpu(TransposeType::NoTranspose, TransposeType::NoTranspose,
        d_grad_outcol, filter, d_grad_input_col,
        batch_size * col_h, col_w, out_channels,
        1.0f, 0.0f, stream);
    // ensure GEMM finished before using d_grad_input_col on the same stream
    // cudaStreamSynchronize(stream); // Removed: MemoryPool handles reuse, no need to sync if we don't free immediately to OS

    // 5) col2im: accumulate grad_input_col -> grad_input (NCHW)
    col2im(d_grad_input_col, grad_input, batch_size, in_channels, height, width, stream);

    // 6) free temporaries
    MemoryPool::instance().deallocate(d_grad_input_col, grad_input_col_size);
    MemoryPool::instance().deallocate(d_input_col, input_col_size);
    MemoryPool::instance().deallocate(d_grad_outcol, grad_outcol_size);
}

// ===========Maxpool=============

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

// ===========Softmax=============

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

// ===========Crossentropy=============

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

__global__ void loss_kernel(const float* input, const float* labels, float* loss,
    int batch_size, int num_classes){
    int row = blockIdx.x;
    if (row >= batch_size) return;

    if (threadIdx.x == 0) {
        int label = int(labels[row]);
        float prob = input[row * num_classes + label];
        prob = fmaxf(prob, 1e-12f);  // 防止 log(0)
        float local_loss = -logf(prob);
        atomicAdd(loss, local_loss);
    }
}

void forward_cross_entropy(const float* input, const float* labels, float* loss,
    int batch_size, int num_classes, cudaStream_t stream){
    // input: logits (batch_size, num_classes)
    // labels: (batch_size)
    // loss: pointer to single float (output)

    size_t softmax_size = (size_t)batch_size * num_classes * sizeof(float);
    float* d_softmax = (float*)MemoryPool::instance().allocate(softmax_size);

    // logits -> softmax
    forward_softmax(input, d_softmax, batch_size, num_classes, stream);

    float h_zero = 0.0f;
    size_t loss_size = sizeof(float);
    float* d_loss = (float*)MemoryPool::instance().allocate(loss_size);
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

    MemoryPool::instance().deallocate(d_loss, loss_size);
    MemoryPool::instance().deallocate(d_softmax, softmax_size);
}

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

__global__ void scale_grad(float* grad_input, int n, float scale){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride){
        grad_input[i] *= scale;
    }
}

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

// ===========SGC=============
__global__ void sgd_update_kernel(float* param, const float* grad, float* velocity,
                                  float lr, float momentum, float weight_decay, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx];
        
        // Apply weight decay first
        if (weight_decay != 0.0f) {
            g += param[idx] * weight_decay;
        }
        
        if (momentum != 0.0f) {
            // Update velocity: v = v * momentum + g
            float v_old = velocity[idx];
            float v_new = v_old * momentum + g;
            velocity[idx] = v_new;
            // Update parameter: param -= lr * v
            param[idx] -= lr * v_new;
        } else {
            // No momentum: param -= lr * g
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

// ===========Batch SGD=============
__global__ void batch_sgd_update_kernel(float* params, const float* grads, float* velocities,
                                        float lr, float momentum, float weight_decay, int total_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        float g = grads[idx];
        
        // Apply weight decay first
        if (weight_decay != 0.0f) {
            g += params[idx] * weight_decay;
        }
        
        if (momentum != 0.0f && velocities != nullptr) {
            // Update velocity: v = v * momentum + g
            float v_old = velocities[idx];
            float v_new = v_old * momentum + g;
            velocities[idx] = v_new;
            // Update parameter: param -= lr * v
            params[idx] -= lr * v_new;
        } else {
            // No momentum: param -= lr * g
            params[idx] -= lr * g;
        }
    }
}

// ===========Fused Operations=============

// Fused Conv2D + ReLU forward function
void conv2d_relu_forward_gpu(const float* input, const float* filter, const float* bias, float* output,
                            int batch_size, int out_channels, int in_channels,
                            int height, int width, int kernel_size, int stride, int padding,
                            cudaStream_t stream) {
    static cudnnHandle_t cudnn = nullptr;
    static cudnnTensorDescriptor_t input_desc = nullptr;
    static cudnnFilterDescriptor_t filter_desc = nullptr;
    static cudnnConvolutionDescriptor_t conv_desc = nullptr;
    static cudnnTensorDescriptor_t output_desc = nullptr;
    static cudnnActivationDescriptor_t activation_desc = nullptr;
    static cudnnTensorDescriptor_t bias_desc = nullptr;

    if (cudnn == nullptr) {
        cudnnCreate(&cudnn);
        cudnnCreateTensorDescriptor(&input_desc);
        cudnnCreateFilterDescriptor(&filter_desc);
        cudnnCreateConvolutionDescriptor(&conv_desc);
        cudnnCreateTensorDescriptor(&output_desc);
        cudnnCreateActivationDescriptor(&activation_desc);
        cudnnCreateTensorDescriptor(&bias_desc);
    }
    cudnnSetStream(cudnn, stream);

    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batch_size, in_channels, height, width);

    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               out_channels, in_channels, kernel_size, kernel_size);

    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Enable Tensor Cores
    cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);

    int out_n, out_c, out_h, out_w;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
                                          &out_n, &out_c, &out_h, &out_w);

    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               out_n, out_c, out_h, out_w);

    // Activation Descriptor for ReLU
    cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

    // Find best algorithm with caching
    using AlgoKey = std::tuple<int, int, int, int, int, int, int>;
    static std::map<AlgoKey, cudnnConvolutionFwdAlgo_t> algo_cache;
    AlgoKey key = std::make_tuple(batch_size, out_channels, in_channels, height, width, kernel_size, stride);
    
    cudnnConvolutionFwdAlgo_t algo;
    auto it = algo_cache.find(key);
    if (it != algo_cache.end()) {
        algo = it->second;
    } else {
        int requestedAlgoCount = 1;
        int returnedAlgoCount;
        cudnnConvolutionFwdAlgoPerf_t perfResults;
        cudnnGetConvolutionForwardAlgorithm_v7(cudnn, input_desc, filter_desc, conv_desc, output_desc,
                                               requestedAlgoCount, &returnedAlgoCount, &perfResults);
        algo = perfResults.algo;
        algo_cache[key] = algo;
    }

    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size);

    void* workspace = nullptr;
    if (workspace_size > 0) {
        workspace = MemoryPool::instance().allocate(workspace_size);
    }

    float alpha = 1.0f, beta = 0.0f;
    
    // Bias
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1);
    
    const float* d_bias = bias;
    bool allocated_bias = false;
    if (d_bias == nullptr) {
        d_bias = (float*)MemoryPool::instance().allocate(out_channels * sizeof(float));
        cudaMemsetAsync((void*)d_bias, 0, out_channels * sizeof(float), stream);
        allocated_bias = true;
    }

    cudnnConvolutionBiasActivationForward(cudnn,
                                          &alpha, input_desc, input,
                                          filter_desc, filter,
                                          conv_desc, algo, workspace, workspace_size,
                                          &beta, output_desc, output,
                                          bias_desc, d_bias,
                                          activation_desc, output_desc, output);

    if (workspace_size > 0) {
        MemoryPool::instance().deallocate(workspace, workspace_size);
    }
    if (allocated_bias) {
        MemoryPool::instance().deallocate((void*)d_bias, out_channels * sizeof(float));
    }
}

__global__ void nchw_to_nhwc_relu_mask_kernel(const float* bchw, const float* output, float* bhwc,
                                      int batch, int out_channels, int height, int width){
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * out_channels * height * width;
    if (idx >= total) return;

    // idx corresponds to NCHW layout
    float val = bchw[idx];
    
    // Apply ReLU mask: if output <= 0, gradient is 0
    if (output[idx] <= 0.0f) {
        val = 0.0f;
    }

    // 依据 NCHW 线性布局恢复坐标： idx = (((b * C + oc) * H) + h) * W + w
    int w = idx % width;
    size_t t = idx / width;
    int h = t % height;
    t = t / height;
    int oc = t % out_channels;
    int b = t / out_channels;

    int col_h = height * width;
    int outcol_idx = (b * col_h + h * width + w) * out_channels + oc;
    bhwc[outcol_idx] = val;
}

// Fused Conv2D + ReLU backward function
void conv2d_relu_backward_gpu(const float* grad_output, const float* output, 
                             const float* input, const float* filter,
                             float* grad_input, float* grad_filter, float* grad_bias,
                             int batch_size, int out_channels, int in_channels,
                             int height, int width, int kernel_size, int stride, int padding,
                             cudaStream_t stream = 0) {
    static cudnnHandle_t cudnn = nullptr;
    static cudnnTensorDescriptor_t x_desc = nullptr;
    static cudnnTensorDescriptor_t y_desc = nullptr;
    static cudnnTensorDescriptor_t dx_desc = nullptr;
    static cudnnTensorDescriptor_t dy_desc = nullptr;
    static cudnnFilterDescriptor_t w_desc = nullptr;
    static cudnnFilterDescriptor_t dw_desc = nullptr;
    static cudnnConvolutionDescriptor_t conv_desc = nullptr;
    static cudnnActivationDescriptor_t act_desc = nullptr;
    static cudnnTensorDescriptor_t db_desc = nullptr;

    if (cudnn == nullptr) {
        cudnnCreate(&cudnn);
        cudnnCreateTensorDescriptor(&x_desc);
        cudnnCreateTensorDescriptor(&y_desc);
        cudnnCreateTensorDescriptor(&dx_desc);
        cudnnCreateTensorDescriptor(&dy_desc);
        cudnnCreateFilterDescriptor(&w_desc);
        cudnnCreateFilterDescriptor(&dw_desc);
        cudnnCreateConvolutionDescriptor(&conv_desc);
        cudnnCreateActivationDescriptor(&act_desc);
        cudnnCreateTensorDescriptor(&db_desc);
    }
    cudnnSetStream(cudnn, stream);

    // Set descriptors
    // Input X: (N, C_in, H, W)
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batch_size, in_channels, height, width);
    cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               batch_size, in_channels, height, width);

    // Filter W: (C_out, C_in, K, K)
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               out_channels, in_channels, kernel_size, kernel_size);
    cudnnSetFilter4dDescriptor(dw_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               out_channels, in_channels, kernel_size, kernel_size);

    // Convolution
    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);

    // Output Y dimensions
    int out_n, out_c, out_h, out_w;
    cudnnGetConvolution2dForwardOutputDim(conv_desc, x_desc, w_desc,
                                          &out_n, &out_c, &out_h, &out_w);

    // Output Y: (N, C_out, H_out, W_out)
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               out_n, out_c, out_h, out_w);
    cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               out_n, out_c, out_h, out_w);

    // Activation
    cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

    // 1. Backward Activation (ReLU)
    // We need a temporary buffer for d_conv_output (gradient before ReLU)
    // d_conv_output has same shape as Y
    size_t y_size = out_n * out_c * out_h * out_w * sizeof(float);
    float* d_conv_output = (float*)MemoryPool::instance().allocate(y_size);

    float alpha = 1.0f, beta = 0.0f;
    // cudnnActivationBackward(handle, actDesc, alpha, yDesc, y, dyDesc, dy, xDesc, x, beta, dxDesc, dx)
    // Here: y=output, dy=grad_output, x=output (safe for ReLU), dx=d_conv_output
    cudnnActivationBackward(cudnn, act_desc, &alpha,
                            y_desc, output,
                            dy_desc, grad_output,
                            y_desc, output, // x = y for ReLU
                            &beta,
                            dy_desc, d_conv_output);

    // Caching setup
    using AlgoKey = std::tuple<int, int, int, int, int, int, int>;
    static std::map<AlgoKey, cudnnConvolutionBwdDataAlgo_t> data_algo_cache;
    static std::map<AlgoKey, cudnnConvolutionBwdFilterAlgo_t> filter_algo_cache;
    AlgoKey key = std::make_tuple(batch_size, out_channels, in_channels, height, width, kernel_size, stride);

    // 2. Backward Data (dX)
    cudnnConvolutionBwdDataAlgo_t data_algo;
    auto it_data = data_algo_cache.find(key);
    if (it_data != data_algo_cache.end()) {
        data_algo = it_data->second;
    } else {
        int returnedAlgoCount;
        cudnnConvolutionBwdDataAlgoPerf_t data_perfResults;
        cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn, w_desc, dy_desc, conv_desc, dx_desc,
                                                    1, &returnedAlgoCount, &data_perfResults);
        data_algo = data_perfResults.algo;
        data_algo_cache[key] = data_algo;
    }

    size_t data_workspace_size = 0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, w_desc, dy_desc, conv_desc, dx_desc, data_algo, &data_workspace_size);
    
    void* data_workspace = nullptr;
    if (data_workspace_size > 0) {
        data_workspace = MemoryPool::instance().allocate(data_workspace_size);
    }

    cudnnConvolutionBackwardData(cudnn, &alpha,
                                 w_desc, filter,
                                 dy_desc, d_conv_output,
                                 conv_desc, data_algo, data_workspace, data_workspace_size,
                                 &beta, dx_desc, grad_input);

    if (data_workspace_size > 0) {
        MemoryPool::instance().deallocate(data_workspace, data_workspace_size);
    }

    // 3. Backward Filter (dW)
    cudnnConvolutionBwdFilterAlgo_t filter_algo;
    auto it_filter = filter_algo_cache.find(key);
    if (it_filter != filter_algo_cache.end()) {
        filter_algo = it_filter->second;
    } else {
        int returnedAlgoCount;
        cudnnConvolutionBwdFilterAlgoPerf_t filter_perfResults;
        cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn, x_desc, dy_desc, conv_desc, dw_desc,
                                                      1, &returnedAlgoCount, &filter_perfResults);
        filter_algo = filter_perfResults.algo;
        filter_algo_cache[key] = filter_algo;
    }

    size_t filter_workspace_size = 0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn, x_desc, dy_desc, conv_desc, dw_desc, filter_algo, &filter_workspace_size);

    void* filter_workspace = nullptr;
    if (filter_workspace_size > 0) {
        filter_workspace = MemoryPool::instance().allocate(filter_workspace_size);
    }

    cudnnConvolutionBackwardFilter(cudnn, &alpha,
                                   x_desc, input,
                                   dy_desc, d_conv_output,
                                   conv_desc, filter_algo, filter_workspace, filter_workspace_size,
                                   &beta, dw_desc, grad_filter);

    if (filter_workspace_size > 0) {
        MemoryPool::instance().deallocate(filter_workspace, filter_workspace_size);
    }

    // 4. Backward Bias (db)
    if (grad_bias != nullptr) {
        cudnnSetTensor4dDescriptor(db_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_channels, 1, 1);
        cudnnConvolutionBackwardBias(cudnn, &alpha,
                                     dy_desc, d_conv_output,
                                     &beta, db_desc, grad_bias);
    }

    // Cleanup
    MemoryPool::instance().deallocate(d_conv_output, y_size);
}

// ===========Adam=============
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

void batch_sgd_update_gpu(float* params, const float* grads, float* velocities,
                          float lr, float momentum, float weight_decay, int total_size, cudaStream_t stream) {
    int blockSize = 256;
    int gridSize = (total_size + blockSize - 1) / blockSize;
    batch_sgd_update_kernel<<<gridSize, blockSize, 0, stream>>>(params, grads, velocities, lr, momentum, weight_decay, total_size);
}

void adam_update_gpu(float* param, const float* grad, float* m, float* v,
                     float lr, float beta1, float beta2, float eps, float weight_decay, 
                     int step, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    adam_update_kernel<<<blocks, threads, 0, stream>>>(param, grad, m, v, lr, beta1, beta2, eps, weight_decay, step, size);
}

// ===========Ewise Ops=============
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

// ===========Scalar Ops=============
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

// ===========Broadcast=============
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

// ===========Batchnorm=============

// Pass 1: Compute Mean
__global__ void batch_norm_collect_mean_kernel(
    const float* input, float* mean,
    int batch_size, int channels, int height, int width) {
    
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int spatial_size = height * width;
    int num_elements = batch_size * spatial_size;
    
    float sum = 0.0f;
    
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / spatial_size;
        int hw = i % spatial_size;
        int idx = n * channels * spatial_size + c * spatial_size + hw;
        sum += input[idx];
    }
    
    __shared__ float s_sum[256];
    s_sum[tid] = sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        mean[c] = s_sum[0] / num_elements;
    }
}

// Pass 2: Compute Variance
__global__ void batch_norm_collect_variance_kernel(
    const float* input, const float* mean, float* var,
    int batch_size, int channels, int height, int width) {
    
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int spatial_size = height * width;
    int num_elements = batch_size * spatial_size;
    
    float m = mean[c];
    float sum_sq_diff = 0.0f;
    
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / spatial_size;
        int hw = i % spatial_size;
        int idx = n * channels * spatial_size + c * spatial_size + hw;
        float val = input[idx];
        float diff = val - m;
        sum_sq_diff += diff * diff;
    }
    
    __shared__ float s_sum[256];
    s_sum[tid] = sum_sq_diff;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        var[c] = s_sum[0] / num_elements;
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
    float momentum, int channels, int N) {
    
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < channels) {
        float unbiased_var = current_var[c] * N / (float)(N - 1);
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * current_mean[c];
        running_var[c] = (1.0f - momentum) * running_var[c] + momentum * unbiased_var;
    }
}

void batch_norm_forward_training(
    float* input, float* output, 
    float* weight, float* bias,
    float* running_mean, float* running_var,
    float* save_mean, float* save_inv_std,
    int batch_size, int channels, int height, int width,
    float momentum, float eps) {
    
    size_t size = channels * sizeof(float);
    float *d_mean = (float*)MemoryPool::instance().allocate(size);
    float *d_var = (float*)MemoryPool::instance().allocate(size);
    
    // Two-pass algorithm for numerical stability
    batch_norm_collect_mean_kernel<<<channels, 256>>>(
        input, d_mean, batch_size, channels, height, width);
        
    batch_norm_collect_variance_kernel<<<channels, 256>>>(
        input, d_mean, d_var, batch_size, channels, height, width);
        
    batch_norm_forward_kernel<<<(batch_size * channels * height * width + 255) / 256, 256>>>(
        input, output, d_mean, d_var, weight, bias, batch_size, channels, height, width, eps);
        
    batch_norm_save_stats_kernel<<<(channels + 255) / 256, 256>>>(
        d_mean, d_var, save_mean, save_inv_std, channels, eps);
        
    int num_elements = batch_size * height * width;
    update_running_stats_kernel<<<(channels + 255) / 256, 256>>>(
        running_mean, running_var, d_mean, d_var, momentum, channels, num_elements);
        
    // Debug: Add sync to rule out race conditions
    cudaDeviceSynchronize();

    MemoryPool::instance().deallocate(d_mean, size);
    MemoryPool::instance().deallocate(d_var, size);
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
    
    
    batch_norm_backward_reduce_kernel<<<channels, 256>>>(
        grad_output, input, save_mean, save_inv_std, grad_weight, grad_bias,
        batch_size, channels, height, width);
        
    batch_norm_backward_input_kernel<<<(batch_size * channels * height * width + 255) / 256, 256>>>(
        grad_output, input, grad_input, save_mean, save_inv_std, weight,
        grad_bias, grad_weight, batch_size, channels, height, width);
}

// ===========Dropout=============

#include <curand.h>

void fill_random_uniform(float* data, int size, unsigned long long seed) {
    curandGenerator_t gen;
    curandStatus_t status;
    
    status = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    if (status != CURAND_STATUS_SUCCESS) {
        return;
    }
    
    status = curandSetPseudoRandomGeneratorSeed(gen, seed);
    if (status != CURAND_STATUS_SUCCESS) {
        curandDestroyGenerator(gen);
        return;
    }
    
    status = curandGenerateUniform(gen, data, size);
    if (status != CURAND_STATUS_SUCCESS) {
        curandDestroyGenerator(gen);
        return;
    }
    
    curandDestroyGenerator(gen);
}

// out = in * mask * scale
// mask = (rand < prob) ? 1 : 0
// scale = 1 / prob
__global__ void dropout_forward_kernel(const float* in, const float* rand, float* out, float* mask, 
                                       int size, float prob, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float r = rand[idx];
        float m = (r < prob) ? 1.0f : 0.0f;
        mask[idx] = m;
        out[idx] = in[idx] * m * scale;
    }
}

void dropout_forward(const float* in, float* out, float* mask, float* rand, 
                     int size, float prob, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    float scale = 1.0f / prob;
    dropout_forward_kernel<<<blocks, threads, 0, stream>>>(in, rand, out, mask, size, prob, scale);
}

// grad_in = grad_out * mask * scale
__global__ void dropout_backward_kernel(const float* grad_out, const float* mask, float* grad_in, 
                                        int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_in[idx] = grad_out[idx] * mask[idx] * scale;
    }
}

void dropout_backward(const float* grad_out, const float* mask, float* grad_in, 
                      int size, float prob, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    float scale = 1.0f / prob;
    dropout_backward_kernel<<<blocks, threads, 0, stream>>>(grad_out, mask, grad_in, size, scale);
}

__global__ void crop_kernel(const float* input, float* output, const int* crop_top, const int* crop_left,
                            int batch, int channels, int height, int width, int padding, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int w = idx % width;
    int temp = idx / width;
    int h = temp % height;
    temp = temp / height;
    int c = temp % channels;
    int n = temp / channels;

    int top = crop_top[n];
    int left = crop_left[n];

    int in_y = h + top - padding;
    int in_x = w + left - padding;

    if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
        int in_idx = ((n * channels + c) * height + in_y) * width + in_x;
        output[idx] = input[in_idx];
    } else {
        output[idx] = 0.0f;
    }
}

void crop_gpu(const float* input, float* output, const int* crop_top, const int* crop_left,
              int batch, int channels, int height, int width, int padding, int count) {
    int threads = 1024;
    int blocks = (count + threads - 1) / threads;
    crop_kernel<<<blocks, threads>>>(input, output, crop_top, crop_left, batch, channels, height, width, padding, count);
}

__global__ void flip_kernel(const float* input, float* output, const int* flip_mask,
                            int batch, int channels, int height, int width, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int w = idx % width;
    int temp = idx / width;
    int h = temp % height;
    temp = temp / height;
    int c = temp % channels;
    int n = temp / channels;

    if (flip_mask[n]) {
        int in_x = width - 1 - w;
        int in_idx = ((n * channels + c) * height + h) * width + in_x;
        output[idx] = input[in_idx];
    } else {
        output[idx] = input[idx];
    }
}

void flip_gpu(const float* input, float* output, const int* flip_mask,
              int batch, int channels, int height, int width, int count) {
    int threads = 1024;
    int blocks = (count + threads - 1) / threads;
    flip_kernel<<<blocks, threads>>>(input, output, flip_mask, batch, channels, height, width, count);
}
