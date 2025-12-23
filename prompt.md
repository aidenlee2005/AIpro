Slide 3 — UML 类图（UML Class Diagram）

Prompt for nanobanana:
Create a clean, minimalist UML-style class diagram on a pure white background (16:9). Use a high-end, simple aesthetic with three harmonious colors: (1) Background fill #DEEEEE with border #54A19D for data structures, (2) Background fill #F9F2E7 with border #E29246 for model/module classes, (3) Background fill #ECEDF3 with border #25385D for backend/system components. Layout should fill the space generously with many boxes to avoid excessive whitespace, employing larger sans-serif fonts (e.g., Inter Bold or Helvetica, size 16-18pt for readability) and thin directional arrows. Incorporate subtle shadows under boxes for a slight 3D effect and depth.

Include the following classes and relationships (use clear boxes + thin directional arrows and small annotations, focusing on general relationships without excessive detail):
- `Tensor` (C++/pybind): wraps device pointer and host copy, holds shape, dtype, requires_grad; composition with MemoryPool.
- `MemoryPool`: singleton for memory management; allocates/deallocates to reduce fragmentation.
- `Module` (framework/optimizer.py): contains parameters (list[Tensor]) and children (submodules); composition with Tensor.
- `Optimizer` (framework/optimizer.py): reads/writes Tensor params; uses strategies like SGD/Adam.
- `Operator` / `CudaKernel` / `ConvReLU` (csrc kernels & fused ops): specialized kernels for operations.
- `pybind` interface (py.* functions): boundary between Python and C++/CUDA.
- External: `cuDNN` with key APIs and features.

Visual hints: Assign colors as above. Include small icons for CPU/GPU next to relevant classes. Add subtle shadows under boxes for depth. Fill the canvas with interconnected boxes, arrows, and annotations; ensure clarity with larger fonts and grouped sections.

Deliverable: vector-style SVG with labeled boxes and arrows, exportable PNG. Fill the canvas with interconnected boxes, arrows, and annotations; ensure clarity with larger fonts and grouped sections.

---

Slide 4 — 模块调用流程（Module Call Flow / Data & Control Flow）

Prompt for nanobanana:
Create a clear, logical flow diagram on white background (16:9) to illustrate the module call process and training logic. Use a simple horizontal layout with three main sections, high contrast for clarity. Color palette: (1) #DEEEEE with border #54A19D for data/memory, (2) #F9F2E7 with border #E29246 for Python/frontend, (3) #ECEDF3 with border #25385D for CUDA/backend. Bold sans-serif fonts (size 18-20pt) for readability.

Layout:
- **Left Section (CPU, Color 2)**: Box with "User Code: Define Model & Run Training". Include code snippets: `model = nn.Sequential(...)`; `output = model(input)`; `loss.backward()`; `optimizer.step()`. Annotate: "CPU controls the process."
- **Middle Section (Graph, Color 1)**: Conceptual graph with nodes (Ops) and edges (Tensors). Annotate: "Dynamic graph built during forward."
- **Right Section (GPU, Color 3)**: Box with "CUDA Execution: Kernels compute ops". Annotate: "GPU handles heavy computations."

Arrows & Logic:
- Solid arrows for forward: CPU -> Graph -> GPU (model def -> forward build graph -> compute on GPU).
- Dotted arrows for backward: GPU -> Graph -> CPU (grads from GPU -> accumulate in graph -> update on CPU).
- Highlight: "Backward involves topological sort for reverse order traversal."

Annotations:
- Callouts: "Forward: CPU initiates, GPU computes." "Backward: Reverse graph with topo sort, accumulate grads."
- Code backgrounds: Slightly lighter than box fill.

Visual Style: Rounded boxes with moderately thick borders, soft shadows. Arrows moderately thick, labeled. Simple and logical; maximize clarity.

Deliverable: SVG with layers; export 16:9 PNG for slides.

---

Slide 5 — 实现历程与性能演进（Timeline: Features vs Performance）

Prompt for nanobanana:
Create a two-lane horizontal timeline on white background (16:9). Style: premium minimal, use three tones: teal for feature/functional milestones, orange for performance/optimization milestones, navy for dates/anchors. Use clean icons (code, GPU, stopwatch, test check) and a clear progression left-to-right.

Content (show boxes with date, short title, one-line detail and optional perf number):
Feature lane (top, teal):
- 12.11 — `Tensor` migration & Python Conv/Pool/ReLU implementations (功能实现)
- 12.12 — `Module` + `Optimizer` + forward/backward (可跑两层CNN)
- 12.15 — `BatchNorm2d`, `Dropout` 实现与测试
- 12.17 — 标准模型：VGG / ResNet 部署与验证

Optimization lane (bottom, orange):
- 12.14 — CUDA 算子下沉（底层算子：add/mul/broadcast -> CUDA kernels） — perf: ~110s -> ~19s/ep
- 12.14 — MemoryPool 实现（显存复用，减少 cudaMalloc） — perf: ~19s -> ~12s/ep
- 12.20 — 算子融合（Conv+ReLU）与内核级优化 — perf: ~12s -> ~9s/ep
- 12.22 — cuDNN 集成（cudnnConvolutionBiasActivationForward 等） — perf: ~9s -> ~4s/ep; note algorithm cache & Tensor Core enable

Extras and callouts:
- Add small checkpoint badge for tests: `tests/test_cudnn_conv_relu.py` near cuDNN box and note: “elementwise parity vs PyTorch, error <= 1e-4”.
- Add a vertical performance bar or sparkline above the timeline showing steady reduction in epoch time (110 → 19 → 12 → 9 → 4). Use a smooth curve or stepped bars in navy.
- Add a “next steps” mini-box at the right edge (FP16 mixed precision, DDP/distributed, unit/regression tests).

Visual hints: prefer simple rounded rectangles for milestones, small icons to indicate type (feature vs perf), and subtle connecting dotted lines between related feature/perf pairs (e.g., MemoryPool box connected to CUDA kernels box). Keep typography large enough for slide readability.

Deliverable: SVG + 16:9 PNG. Provide separate labeled layers: `features`, `optimizations`, `perf-graph` for easy editing.

Notes for the generator:
- Use English labels for API names and filenames (e.g., `csrc/tensor.h`, `framework/optimizer.py`, `tests/test_cudnn_conv_relu.py`) so I can cross-reference with the repo.
- Keep palette limited and consistent across three slides so they form a coherent set.

---

End of prompts. Paste each prompt individually into nanobanana to generate the three vector graphics. Recommended output format: SVG (editable) + PNG (presentation-ready).
