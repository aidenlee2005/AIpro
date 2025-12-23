Slide 1 — 项目概述（Project Overview）

Prompt for nanobanana:
Design an introductory slide for a PPT presentation on a pure white background (16:9), optimized as the first slide to set the stage. Use a clean, professional layout with a large centered title at the top, followed by structured sections below to fill the space generously without clutter. Employ larger sans-serif fonts (e.g., Inter Bold or Helvetica, title size 24-28pt, body 16-18pt) for high readability. Incorporate subtle shadows under elements for a subtle 3D depth. Use the specified color palette: (1) Background fill #DEEEEE with border #54A19D; (2) Background fill #F9F2E7 with border #EB8E35; (3) Background fill #ECEDF3 with border #25385D. Add small thematic icons (e.g., calendar, gear, stack) to enhance visual appeal.

Layout Structure:
- **Top Section**: Large centered title "Project Overview" in bold, with a subtitle or tagline like "From Pure Python to GPU-Accelerated Deep Learning Framework Implementation".
- **Middle Section**: Arrange 3-4 rounded rectangular boxes in a row or grid below the title, each representing a key aspect, with icons inside the boxes and concise text. For better visual appeal, arrange text within each box as follows: bold title at the top center, icon positioned beside the title or at the top, and content below in short, readable lines or bullet points.
  - Box 1 (Color 1): **Implementation Timeline** - Icon: Calendar. Content: 2025.12.11 - 2025.12.22.
  - Box 2 (Color 2): **Main Objectives** - Icon: Target. Content: Gradually migrate from pure Python to GPU, integrate cuDNN, balance functionality and performance.
  - Box 3 (Color 3): **Key Modules** - Icon: Gear. Content: Tensor, Autodiff, Operators (CPU/CUDA), Module/Model, Optimizer, MemoryPool, pybind & CMake.
  - Box 4 (Color 1): **Tech Stack** - Icon: Stack. Content: Python, C++/CUDA, pybind11, cuDNN, CMake, Numpy, PyTorch.
- **Bottom Section**: Optional footer with project name or presenter info, but keep minimal.

Visual Hints: Ensure the layout is balanced and engaging for an introduction, with ample white space around elements but filling the canvas. Use thin borders on boxes, soft drop shadows for depth, and simple icons. Typography should be prominent and easy to read from a distance.

Deliverable: SVG with layers (title, boxes, icons) for easy editing; export 16:9 PNG optimized for slides.

---

**Slides 1 — 项目概述**
简短介绍项目背景、实现时间线与主要技术栈：
- **实现时间范围**：2025.12.11-2025.12.22
- **主要目标**：从纯Python实现逐步向 GPU 下沉，最后集成 cuDNN，兼顾功能完整性与训练性能。
- **主要模块**：`Tensor`（张量封装）、自动求导（autodiff）、算子实现（CPU/Python / CUDA）、`Module`/模型实现、`Optimizer`、显存池（MemoryPool）、pybind 接口与 CMake 构建。
- **技术栈**：Python、C++/CUDA、pybind11、cuDNN、CMake、Numpy（数据预处理）、PyTorch（基准对比）。

**Slides 2 — 性能与准确率对比（最终版）**
- 展示与 PyTorch（或官方实现）的对比图表：训练时间 / epoch、训练曲线（loss/accuracy）
- 建议可视化：CNN、CNN+aug、VGG+aug、ResNet+aug 的训练时间对比柱状图；准确率收敛曲线折线图。
- 说明：最终优化后（集成 cuDNN）训练时间显著下降；准确率方面应与 PyTorch 保持可比（如存在差异需检查实现细节与超参）。

**Slides 3 — 实现架构细节：UML 类图（要点）**
- 核心类：`Tensor`（封装数据与梯度）、`Operator`（算子抽象）、`Module`（模型父类）、`Optimizer`、`MemoryPool`、`CudaKernel`/`CudaLauncher`、`DataLoader`。
- 关系说明：`Module` 持有子模块与参数（`Tensor`），前向调用算子，反向调用算子对应的梯度计算；`Optimizer` 负责参数更新并访问 `MemoryPool`/CUDA 资源。

**Slides 4 — 模块调用流程**
- 数据流：DataLoader -> model.forward(inputs) -> loss -> backward() -> optimizer.step()
- 计算流：Python 层封装算子调用 -> pybind 调用 C++/CUDA 实现 -> 在 `MemoryPool` 中分配临时张量 -> 调用 CUDA kernel 执行计算 / cuDNN 接口
- 调试点：tensor 的 device/shape 检查、内存泄露与同步（cudaDeviceSynchronize）、算子数值稳定性（BN、Dropout 的训练/评估切换）。

**Slides 5 — 实现历程（功能实现 & 性能优化）**
（按时间顺序，示例日期保留并修正错别字）
- 12.11：基于 homework5/6，将内部数据从 `np.ndarray` 替换为 `Tensor`，并在 Python 端实现 `Conv2d`、`MaxPool2d`、`ReLU` 等算子（功能实现）。
- 12.12：实现 `Optimizer` 和 `Module` 类，完成前向/反向传播（能跑一个两层卷积的 CNN，约 110s/epoch，示例值）。
- 12.14：将基础算子（加减乘除、广播等）迁移到 CUDA 实现，训练时间下降到约 19s/epoch（性能优化）。
- 12.14（同日）：实现显存池（MemoryPool），减少频繁的 `cudaMalloc`/`cudaFree`，训练时间进一步下降到约 12s/epoch（性能优化）。
- 12.15：实现 `BatchNorm2d`、`Dropout` 等模块，并测试其正确性与性能（功能实现）。
- 12.17：实现 VGG、ResNet 等标准模型并跑通训练与评估（功能实现）。
- 12.20：算子融合（如 Conv2d+ReLU）与内核级优化，训练时间下降到约 9s/epoch（性能优化）。
- 12.22：集成 cuDNN，用高效库替换部分手写卷积算子，训练时间降至约 4s/epoch（性能优化）。

**Slides 6 — 制作感受与下一步**
- 感受要点：
	1) 通过实际优化直观看到性能提升，工程中每一毫秒都有价值；
	2) 良好软件设计（模块化、接口清晰）能让性能优化更容易插拔；
	3) AI 辅助编程与自动化工具能显著提高开发效率。
- 下一步建议：
	- 引入混合精度（FP16）与自动混合精度，从而减少显存并加速训练；
	- 支持分布式训练（DDP），评估多卡扩展效率；
	- 增加单元测试与数值回归测试，确保算子在重构/优化后数值一致性；
	- 补充更多可视化（训练曲线、profiling flamegraph、内存使用曲线）。
