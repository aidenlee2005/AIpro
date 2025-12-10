cmake_minimum_required(VERSION 3.18)
project(py_tensor_cuda LANGUAGES CXX CUDA)

# ----------------------------------------------
# 基础配置
# ----------------------------------------------
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ----------------------------------------------
# 找到 pybind11 和 CUDA
# ----------------------------------------------
find_package(pybind11 REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# ----------------------------------------------
# CUDA 静态库部分 (例如 layers.cu)
# ----------------------------------------------
add_library(layers_cuda STATIC csrc/layers.cu)
target_include_directories(layers_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
target_compile_options(layers_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC>)

# 链接常用 CUDA 库
target_link_libraries(layers_cuda
    PUBLIC
        CUDA::cudart        # CUDA runtime
        CUDA::cublas        # cuBLAS
        CUDA::curand        # cuRAND
        CUDA::cufft         # cuFFT (可选)
        CUDA::cusolver      # cuSolver (可选)
        CUDA::cusparse      # cuSparse (可选)
)

# ----------------------------------------------
# PyBind11 模块部分
# ----------------------------------------------
pybind11_add_module(py_tensor csrc/pybind_tensor.cpp)
target_include_directories(py_tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc)

# 连接 CUDA 静态库
target_link_libraries(py_tensor PRIVATE layers_cuda)

# ----------------------------------------------
# 输出提示信息
# ----------------------------------------------
message(STATUS "CUDAToolkit include dirs: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit libraries: ${CUDAToolkit_LIBRARIES}")

