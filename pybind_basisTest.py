import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))

import numpy as np
import py_tensor as py
from py_tensor import Tensor

print('Tensor的创建')
t = Tensor([3,4],"gpu")
print('Tensor的打印：.print()')
t.print()

print('Tensor的形状：.shape()')
print(t.shape())

print('Tensor的元素个数：.size()')
print(t.size())

print('Tensor与ndarray的转换')
np = np.random.randn(3,4).astype(np.float32)
print('numpy array:')
print(np)
t2 = Tensor.from_numpy(np)
print('Tensor from numpy:')
t2.print()
print('numpy from Tensor:')
np2 = t2.to_numpy()
print(np2)