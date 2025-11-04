cmake_minimum_required(VERSION 3.18)
project(py_tensor_cuda LANGUAGES C CXX)

find_package(pybind11 REQUIRED)

# 尝试找到 CUDAToolkit（现代 CMake）
find_package(CUDAToolkit QUIET)

# 尝试找到 cuda_runtime.h 的目录（兼容不同系统）
find_path(CUDA_INCLUDE_DIR cuda_runtime.h
    HINTS
      $ENV{CUDA_HOME}
      /usr/local/cuda/include
      /usr/include
)
if (NOT CUDA_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find cuda_runtime.h. Set CUDA_HOME or install CUDA toolkit.")
endif()

# 尝试获取 cudart 库
set(CUDART_LIB "" CACHE PATH "path to cudart library")
if (CUDAToolkit_FOUND AND TARGET CUDAToolkit::cudart)
    set(CUDART_TARGET CUDAToolkit::cudart)
else()
    # 尝试查找 libcudart
    find_library(CUDART_LIB cudart
        PATHS /usr/local/cuda/lib64 /usr/local/cuda/lib /usr/lib64 /usr/lib
        NO_DEFAULT_PATH
    )
endif()

set(CMAKE_CXX_STANDARD 14)

pybind11_add_module(py_tensor pybind_tensor.cpp)
target_include_directories(py_tensor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CUDA_INCLUDE_DIR})

if (DEFINED CUDART_TARGET)
    target_link_libraries(py_tensor PRIVATE ${CUDART_TARGET})
elseif (CUDART_LIB)
    target_link_libraries(py_tensor PRIVATE ${CUDART_LIB})
else()
    message(FATAL_ERROR "Could not find CUDA runtime library (cudart). Try installing CUDA dev package or set CUDART_LIB path.")
endif()