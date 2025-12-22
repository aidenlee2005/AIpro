# 深度学习框架 (Custom Deep Learning Framework)

这是一个基于 Python 前端和 C++/CUDA 后端构建的轻量级深度学习框架。它实现了自动微分、动态计算图、显存池管理以及高性能的 CUDA 算子（包括卷积、矩阵乘法、广播机制等）。

## 1. 环境要求

在运行本项目之前，请确保您的环境满足以下要求：

*   **操作系统**: Linux (推荐 Ubuntu 20.04/22.04)
*   **硬件**: 支持 CUDA 的 NVIDIA GPU
*   **软件依赖**:
    *   **Python**: >= 3.8 (开发环境使用 3.12)
    *   **CUDA Toolkit**: >= 11.0 (开发环境使用 12.8)
    *   **CMake**: >= 3.18 (开发环境使用 3.22)
    *   **C++ 编译器**: 支持 C++14 的编译器 (如 g++ 7+)

### Python 依赖库
请确保安装了以下 Python 库：
```bash
pip install numpy pybind11
# 可选：用于读取 MNIST/CIFAR 数据集或对比测试
pip install torch torchvision 
```

## 2. 编译指南

本框架的核心算子和 Tensor 管理由 C++/CUDA 实现，因此在运行 Python 代码前必须先进行编译。

1.  **创建构建目录**
    ```bash
    mkdir build
    cd build
    ```

2.  **运行 CMake 配置**
    ```bash
    cmake ..
    ```

3.  **编译项目**
    ```bash
    make -j$(nproc)
    ```

编译成功后，会在 `build/` 目录下生成 `py_tensor.cpython-*.so` 共享库文件。Python 前端会自动加载此模块。

## 3. 运行示例

### 训练 CIFAR-10 模型
本项目包含多个 CIFAR-10 训练示例，涵盖了从简单 CNN 到 ResNet 的不同复杂度模型。

1.  **基础 CNN (无数据增强)**
    ```bash
    python examples/train_cnn_no_aug.py
    ```

2.  **基础 CNN (带数据增强)**
    ```bash
    python examples/train_cnn_aug.py
    ```

3.  **VGG 网络**
    ```bash
    python examples/train_vgg_aug.py
    ```

4.  **ResNet 网络**
    ```bash
    python examples/train_resnet_aug.py
    ```

所有脚本运行后，训练日志会自动记录到 `examples/new_training_log.csv`，模型文件会保存到 `examples/models/` 目录。

**运行命令：**
```bash
# 请在项目根目录下运行
python examples/train_cifar10_custom.py
```

*注意：脚本会自动下载 CIFAR-10 数据集到 `data/` 目录（如果尚未下载）。*

### 读取 MNIST 数据
```bash
python examples/mnist_read.py
```

## 4. 运行测试

为了验证框架的正确性，我们提供了一系列单元测试。

### 基础功能测试
测试 Tensor 的创建、内存管理和基本属性。
```bash
python tests/pybind_basisTest.py
```

### 算子正确性测试
测试各个 CUDA 算子（如 Conv2d, MatMul, ReLU, BatchNorm, Dropout 等）的计算结果是否与 PyTorch/Numpy 一致。
```bash
python tests/pybind_opTest.py
python tests/test_dropout_custom.py
```

### 完整网络测试
测试 CNN 网络的前向和反向传播流程。
```bash
python tests/test_cnn_custom.py
```

## 4. 超参数调优
运行以下脚本进行超参数搜索（约30分钟）：
```bash
python examples/tune_cifar10.py
```
该脚本会尝试不同的学习率、Batch Size和Dropout率组合，并输出最优配置。

## 5. 项目结构

```
.
├── csrc/               # C++/CUDA 后端源码 (Tensor, MemoryPool, Kernels)
├── framework/          # Python 前端框架 (Autodiff, Operators, Optimizer)
├── examples/           # 示例脚本 (CIFAR-10 Training)
├── models/             # 模型存储
├── tests/              # 测试脚本
├── data/               # 数据集存放目录
├── docs/               # 文档
├── build/              # 编译输出目录
└── CMakeLists.txt      # CMake 构建配置
```

## 6. 常见问题

*   **ImportError: No module named 'py_tensor'**:
    *   请确保您已经成功执行了编译步骤。
    *   请确保运行脚本时，`build/` 目录在 `PYTHONPATH` 中，或者脚本中包含了正确的 `sys.path.append`（示例脚本已包含）。

*   **CUDA Error**:
    *   请检查 `nvidia-smi` 确认驱动安装正确。
    *   确保编译时使用的 CUDA 版本与运行时一致。
