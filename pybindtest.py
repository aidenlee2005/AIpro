import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import py_tensor
import numpy as np
from py_tensor import Tensor

a = np.random.randn(2,3).astype(np.float32)
t = py_tensor.Tensor.from_numpy(a, "gpu")

arr = t.to_numpy()
print(arr.shape, arr.dtype)
t.print()

t2 = py_tensor.Tensor([2,3],"cpu")
print(t2.is_cpu())