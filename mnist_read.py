import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))

import py_tensor
from py_tensor import Tensor
import numpy as np
from torchvision import datasets, transforms
import torch

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
    # 简单 demo：加载训练集前 100 张并转换为 py_tensor.Tensor
    imgs, labels = load_mnist_via_torch(download=True, train=True)
    print("imgs:", imgs.shape, imgs.dtype, "labels:", labels.shape)

    # 例如只取前 100，转换为你的绑定 Tensor（设备可改为 'gpu'）
    imgs_small = imgs[:100]  # shape (100,1,28,28)
    labels_small = labels[:100]

    try:
        t_imgs = numpy_to_py_tensor(imgs_small, device="cpu")
        t_labels = numpy_to_py_tensor(labels_small.reshape(-1, 1), device="cpu")
        print("转换为 py_tensor.Tensor 成功：", t_imgs.shape(), t_labels.shape())
    except Exception as e:
        print("转换为 py_tensor.Tensor 失败：", e)


