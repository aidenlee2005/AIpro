import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))

import py_tensor as py
from py_tensor import Tensor
import numpy as np
from torchvision import datasets, transforms
import torch
import matplotlib
matplotlib.use('Agg')  # headless 环境使用非交互后端，确保可以保存图片
import matplotlib.pyplot as plt

def load_mnist_via_torch(root="data", train=True, download=True):
    """
    返回 (images, labels)：
      images: np.ndarray, dtype=float32, shape (N, 1, 28, 28), 值在 [0,1]
      labels: np.ndarray, dtype=int64, shape (N,)
    需要 torchvision & torch 可用。
    """
    ds = datasets.MNIST(root=root, train=train, download=download,
                        transform=transforms.ToTensor())  # returns torch.Tensor in [0,1], shape (1,28,28)
    imgs = torch.stack([img for img, _ in ds])  # shape (N,1,28,28)
    labels = np.array([label for _, label in ds], dtype=np.int64)
    return imgs.numpy().astype(np.float32), labels

def numpy_to_py_tensor(arr: np.ndarray, device="gpu"):
    """
    将 numpy 转为 pybind 导出的 Tensor（调用 Tensor.from_numpy）。
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("需要 numpy.ndarray")
    return Tensor.from_numpy(arr.astype(np.float32), device)

if __name__ == "__main__":
    # 选择设备，优先 GPU 否则 CPU
    preferred = "gpu" if torch.cuda.is_available() else "cpu"
    print("torch.cuda.is_available():", torch.cuda.is_available(), " selected:", preferred)

    # 简单 demo：加载训练集前若干张并转换为 py_tensor.Tensor
    imgs, labels = load_mnist_via_torch(download=False, train=True)
    print("imgs:", imgs.shape, imgs.dtype, "labels:", labels.shape)

    imgs_small = imgs[:5]  # 只取前 5 张用于展示
    labels_small = labels[:5]

    # 先尝试在首选设备上转换，失败则回退到 CPU
    try:
        t_imgs = numpy_to_py_tensor(imgs_small, device=preferred)
        t_labels = numpy_to_py_tensor(labels_small.reshape(-1, 1), device=preferred)
        device_used = preferred
        print("转换为 py_tensor.Tensor 成功：", t_imgs.shape(), t_labels.shape(), "device:", device_used)
    except Exception as e:
        print("转换为 py_tensor.Tensor 在", preferred, "失败：", e, "，回退到 cpu")
        t_imgs = numpy_to_py_tensor(imgs_small, device="cpu")
        t_labels = numpy_to_py_tensor(labels_small.reshape(-1, 1), device="cpu")
        device_used = "cpu"

    # 卷积核（确保与输入在同一设备）
    kernal = np.array([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]], dtype=np.float32)
    kernal_t = numpy_to_py_tensor(kernal, device=device_used)

    # 执行卷积，若在 GPU 上出错则回退到 CPU 并重试
    try:
        conv_t = py.conv2d_forward(t_imgs, kernal_t)
    except Exception as e:
        print("conv2d_forward 在", device_used, "出错：", e)
        if device_used == "gpu":
            print("回退到 CPU 重新计算卷积")
            t_imgs = numpy_to_py_tensor(imgs_small, device="cpu")
            kernal_t = numpy_to_py_tensor(kernal, device="cpu")
            conv_t = py.conv2d_forward(t_imgs, kernal_t)
            device_used = "cpu"
        else:
            raise

    print("卷积结果 tensor（设备", device_used, "）:", getattr(conv_t, "shape", lambda: None)())

    # 转为 numpy 并展示（显示第一通道）
    try:
        conv_np = conv_t.to_numpy()
    except Exception as e:
        # 若在 GPU 上无法直接转 numpy，先在 CPU 上重算
        print("conv_t.numpy() 失败：", e)
        if device_used == "gpu":
            t_imgs_cpu = numpy_to_py_tensor(imgs_small, device="cpu")
            kernal_t_cpu = numpy_to_py_tensor(kernal, device="cpu")
            conv_t_cpu = py.conv2d_forward(t_imgs_cpu, kernal_t_cpu)
            conv_np = conv_t_cpu.numpy()
        else:
            raise

    def show_conv_images(conv_array, num_images=5, per_row=5, out_path="conv_result.png"):
        arr = conv_array
        if arr.ndim == 4:
            N, C, H, W = arr.shape
        elif arr.ndim == 3:
            N, H, W = arr.shape
            C = 1
            arr = arr.reshape(N, C, H, W)
        else:
            raise ValueError("不支持的 conv_array 维度: {}".format(arr.shape))

        num = min(num_images, N)
        rows = (num + per_row - 1) // per_row
        plt.figure(figsize=(per_row * 2, rows * 2))
        for i in range(num):
            img = arr[i]
            channel_img = img[0] if img.ndim == 3 else img
            plt.subplot(rows, per_row, i + 1)
            plt.imshow(channel_img, cmap="gray")
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        print("已保存卷积结果到：", os.path.abspath(out_path))
        # headless 环境通常不调用 plt.show()，若本地运行并希望弹窗可取消注释下一行
        # plt.show()

    show_conv_images(conv_np, num_images=5, per_row=5)


