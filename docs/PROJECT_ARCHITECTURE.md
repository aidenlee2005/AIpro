# 深度学习框架架构文档

本文档详细说明了本深度学习框架（基于 Python 前端 + C++/CUDA 后端）的架构设计、文件组织、核心数据结构以及数据流转逻辑。

## 1. 架构概览

本框架采用典型的混合架构设计，旨在结合 Python 的易用性和 C++/CUDA 的高性能。

*   **前端 (Python 层)**: 负责计算图构建、自动微分调度、模型定义、数据加载和训练循环。
*   **接口层 (Pybind11)**: 负责 Python 对象与 C++ 对象之间的转换，将底层高性能算子暴露给 Python。
*   **后端 (C++ 层)**: 负责 Tensor 的元数据管理、内存管理（显存池）和广播逻辑处理。
*   **内核层 (CUDA)**: 负责实际的并行计算（卷积、矩阵乘法、激活函数、优化器更新等）。

---

## 2. 文件结构与功能说明

项目经过重构，采用了模块化的目录结构，清晰分离了后端核心、前端框架、示例和测试代码。

### 2.1 `csrc/` - C++/CUDA 后端核心

此目录存放所有底层高性能实现代码。

| 文件名 | 功能描述 | 核心函数/类 |
| :--- | :--- | :--- |
| **`memory_pool.h`** | **显存管理**。实现了一个单例模式的缓存分配器 (Caching Allocator)。 | `MemoryPool` (类)<br>- `allocate`: 申请显存（优先复用）<br>- `deallocate`: 归还显存到池<br>- `clear`: 清空池 |
| **`tensor.h`** | **Tensor 定义**。定义了 C++ 端的 Tensor 类，管理 Shape、Strides 和数据指针。集成显存池。 | `Tensor<T>` (类)<br>- 构造/析构: 调用 MemoryPool<br>- `data()`: 获取裸指针 |
| **`layers.cu`** | **CUDA 核函数实现**。包含所有算子的 GPU 实现。集成 **cuDNN** 库以加速卷积操作。 | `gemm_gpu`: 矩阵乘法 (cuBLAS)<br>`conv2d_relu_*`: 融合卷积 (cuDNN)<br>`eltwise_*_kernel`: 逐元素操作<br>`*_broadcast_kernel`: 广播操作<br>`sgd/adam_update_gpu`: 优化器更新<br>`batch_norm_*`: 批归一化<br>`dropout_*`: Dropout |
| **`layers.h`** | **C++ 接口声明**。声明 `layers.cu` 中暴露给 C++ 的函数。 | `TensorStrides` (结构体): 用于广播索引<br>`fill_random_uniform`: 随机数生成 |
| **`pybind_tensor.cpp`** | **Python 绑定**。使用 Pybind11 封装 C++ Tensor 和函数，处理广播步长计算。 | `py_tensor` (模块)<br>`compute_broadcast_strides`: 计算广播后的虚拟步长<br>`make_tensor_like`: 创建同型 Tensor<br>`dropout_*_wrapper`: Dropout 封装 |

### 2.2 `framework/` - Python 前端框架

此目录存放构建计算图和自动微分引擎的 Python 代码。

| 文件名 | 功能描述 | 核心函数/类 |
| :--- | :--- | :--- |
| **`operators.py`** | **算子与图节点**。定义了 Python 端的 `Tensor` 类（继承自 `Value`）和各种算子 (`Op`)。 | `Tensor` (类): 包装 C++ Tensor 对象<br>`EWiseAdd`, `MatMul`, `Conv2D`: 具体算子<br>`compute`: 调用 C++ 后端<br>`gradient`: 定义反向传播逻辑<br>`Dropout`: Dropout 算子 |
| **`optimizer.py`** | **模型与优化器**。定义了神经网络层和优化器。 | `Module`, `Parameter`: 模型基类<br>`Linear`, `Conv2d`: 网络层<br>`BatchNorm2d`, `Dropout`: 正则化层<br>`SGD`, `Adam`: 优化器 (调用 C++ `*_step`) |
| **`autodiff.py`** | **自动微分**。实现了反向模式自动微分引擎。 | `compute_gradient_of_variables`: 计算梯度 |
| **`basic_operator.py`** | **基础抽象**。定义了计算图的基本节点类型。 | `Value`: 计算图节点基类<br>`Op`: 算子基类 |
| **`tensor.py`** | **自动微分 Tensor**。扩展了基础 Tensor，支持自动微分。 | `TensorFull`: 支持 `.backward()` 的 Tensor |
| **`device.py`** | **设备抽象**。 | `Device`, `cpu`, `gpu` |
| **`utils.py`** | **工具函数**。 | `rand`, `one_hot` 等 |

### 2.3 `examples/` - 示例脚本

| 文件名 | 功能描述 |
| :--- | :--- |
| **`train_cifar10_custom.py`** | **CIFAR-10 训练**。演示如何使用框架搭建 CNN 并训练 CIFAR-10 数据集。 |
| **`mnist_read.py`** | **MNIST 读取**。辅助脚本，用于读取 MNIST 数据。 |

### 2.4 `tests/` - 测试代码

| 文件名 | 功能描述 |
| :--- | :--- |
| **`pybind_opTest.py`** | **算子单元测试**。测试各个 C++ 算子的正确性（对比 PyTorch/Numpy）。 |
| **`pybind_basisTest.py`** | **基础功能测试**。测试 Tensor 创建、打印、内存管理等。 |
| **`test_cnn_custom.py`** | **网络集成测试**。测试完整 CNN 网络的构建和运行。 |

### 2.5 `docs/` - 文档

| 文件名 | 功能描述 |
| :--- | :--- |
| **`PROJECT_ARCHITECTURE.md`** | **架构文档**。即本文档。 |
| **`更新日志.md`** | **变更记录**。记录项目开发过程中的重要变更。 |

---

## 3. 核心数据结构

### 3.1 Tensor (C++ 端 - `tensor.h`)
这是实际存储数据的对象，驻留在 C++ 堆中，由 `std::shared_ptr` 管理生命周期。
```cpp
template<typename T>
class Tensor {
    std::vector<int> shape;    // 维度信息
    std::vector<int> strides;  // 物理步长
    T* d_data;                 // GPU 显存指针
    Device device;             // 设备类型 (CPU/GPU)
    // ...
};
```
*   **内存管理**: 构造时通过 `MemoryPool::instance().allocate()` 获取显存，析构时通过 `deallocate()` 归还。

### 3.2 Tensor (Python 端 - `operators.py`)
这是用户操作的对象，它是计算图中的一个节点 (`Value`)。
```python
class Tensor(Value):
    def __init__(self, ...):
        # cached_data 是一个 C++ Tensor 对象 (py_tensor.Tensor)
        self.cached_data = ... 
        self.op = ...      # 产生该 Tensor 的算子 (用于构建图)
        self.inputs = ...  # 输入节点列表
```

### 3.3 Op (Python 端)
代表一种计算操作（如加法、卷积）。
*   `compute(...)`: **前向传播**。接收输入 `Tensor` 的 `cached_data` (C++ 对象)，调用 `py_tensor` 模块的函数，返回新的 C++ Tensor。
*   `gradient(...)`: **反向传播**。定义如何根据输出梯度计算输入梯度，通常会创建新的 `Op` 节点。

---

## 4. 数据流动与运行逻辑

### 4.1 z = x + y

以 `z = x + y` 为例：

**前向传播**

1.  **Python 调用**: 用户执行 `z = x + y`。
2.  **算子分发**: 触发 `Tensor.__add__`，创建 `EWiseAdd` 算子节点。
3.  **计算执行 (`compute`)**:
    *   `EWiseAdd.compute` 被调用。
    *   检查 `x` 和 `y` 的形状。
    *   **情况 A (形状相同)**: 调用 `py.eltwise_add(x.data, y.data)`。
    *   **情况 B (需要广播)**: Python 端调用 C++ 的 `eltwise_add`，C++ 端 `compute_broadcast_strides` 计算虚拟步长，然后调用 CUDA Kernel。
4.  **CUDA 执行**:
    *   `layers.cu` 中的 Kernel 启动。
    *   GPU 读取 `x`, `y` 显存，计算结果写入 `z` 的显存（从显存池申请）。
5.  **图构建**: 返回一个新的 Python `Tensor` `z`，其 `op` 指向 `EWiseAdd`，`inputs` 指向 `x` 和 `y`。

**反向传播**

1.  **触发**: 用户调用 `loss.backward()`。
2.  **拓扑排序**: `autodiff.py` 对计算图进行逆拓扑排序。
3.  **梯度计算**:
    *   遍历节点，调用 `node.op.gradient(out_grad, node)`。
    *   例如 `EWiseAdd` 的梯度是 `(out_grad, out_grad)`。
    *   例如 `MatMul` 的梯度涉及 `matmul(out_grad, input.T)`。
4.  **递归执行**: 梯度计算会生成新的计算图节点，这些节点在求值时再次触发前向传播的逻辑（调用 C++ 后端）。

**参数更新**

1.  **收集参数**: `optimizer.step()` 收集所有 `Parameter` 及其 `.grad`。
2.  **下沉 C++**: 将参数列表、梯度列表、动量状态列表一次性传入 C++ 函数（如 `py_tensor.adam_step`）。
3.  **Fused Kernel**: C++ 启动 `adam_update_kernel`，在 GPU 上并行更新所有参数，避免了 Python 循环遍历参数的开销。


### 4.2 Sequential 模型的数据流动

当用户定义一个 `nn.Sequential` 模型（如 `model = nn.Sequential(nn.Conv2d(...), nn.ReLU(), nn.Linear(...))`）并启动前向和反向传播时，数据流动如下：

**前向传播过程**：
1.  **模型调用**: 用户执行 `output = model(input)`，触发 `Sequential.__call__`，进而调用 `Sequential.forward(input)`。
2.  **模块遍历**: `Sequential.forward` 依次遍历 `self.modules` 列表中的每个子模块（如 `Conv2d`、`ReLU`、`Linear`）。
3.  **数据传递**: 输入数据 `input` 作为第一个模块的输入，经过 `module(x)` 调用每个模块的 `forward` 方法：
   - `Conv2d.forward`: 调用 `x.conv2d(self.weight)`，触发 `Conv2D` 算子，数据从 Python 层下沉到 C++ 后端，执行 CUDA 卷积核函数。
   - `ReLU.forward`: 调用 `x.relu()`，触发 `ReLU` 算子，执行逐元素激活。
   - `Linear.forward`: 调用 `x @ self.weight`，触发 `MatMul` 算子，执行矩阵乘法。
4.  **融合优化**: 如果 `Sequential` 检测到 `Conv2d + ReLU` 组合，会自动替换为 `ConvReLU` 模块，调用融合核函数（如 `conv2d_relu_kernel`），减少 Kernel 启动次数。
5.  **输出返回**: 最终输出经过所有模块后返回，形成完整的计算图，每个中间结果都是 `Tensor` 节点。

**反向传播过程**：
1.  **梯度触发**: 用户调用 `output.backward()`，`autodiff.py` 对整个计算图进行逆拓扑排序。
2.  **模块梯度**: 梯度从输出反向流动，通过 `Sequential` 的嵌套结构递归调用每个子模块的梯度计算：
   - 每个模块的 `gradient` 方法（如 `Conv2D.gradient`）生成新的算子节点，计算输入梯度。
   - 数据再次下沉到 C++ 后端，执行相应的反向核函数（如卷积梯度）。
3.  **参数收集**: `Sequential.parameters()` 递归收集所有子模块的参数，用于优化器更新。
4.  **优化器步骤**: `optimizer.step()` 调用 C++ 的融合更新函数（如 `adam_step`），并行更新所有参数。

这种设计支持模块化组合，数据在 Python 计算图和 C++ 后端之间高效流动，实现自动微分和 GPU 加速。

---

## 5. 关键优化技术

1.  **显存池 (Memory Pool)**:
    *   避免了高频的 `cudaMalloc/cudaFree` 系统调用。
    *   通过 `std::map<size_t, vector<void*>>` 复用显存块，显著减少了分配开销。

2.  **C++ 算子下沉**:
    *   所有计算密集型操作（包括简单的加减乘除）全部在 C++ 实现。
    *   Python 仅作为调度器，不接触实际数据。

3.  **广播机制 (Broadcasting)**:
    *   在 C++ 层实现了 Numpy 风格的广播。
    *   通过 `TensorStrides` 结构体在 CUDA Kernel 中进行坐标映射，无需在内存中显式复制数据即可处理 `(N, C) + (C,)` 等操作。

4.  **优化器融合**:
    *   将参数更新逻辑合并为一个 CUDA Kernel，大幅减少了 Kernel Launch 次数和 Python 循环开销。
