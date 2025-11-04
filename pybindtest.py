import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))
import py_tensor
import numpy as np
import torch
from torch.nn import functional as F
from py_tensor import Tensor

input = torch.tensor(
        [[ 0.9, 0.8, 0.5 ],
        [ 0.8, 0.2, 0.6 ],
        [ 0.1, 0.4, 0.4 ],
        [ 0.5, 0.5, 0.2 ],
        [ 0.2, 0.8, 0.8 ]],
    dtype=torch.float32, requires_grad=True)  # 输入形状是 (batch_size, in_features)

weight = torch.tensor(
        [[ 0.9, 0.8 ],
        [ 0.5, 0.8 ],
        [ 0.2, 0.6 ]],
    dtype=torch.float32, requires_grad=True) # 权重形状是 (out_features, in_features)

bias = torch.tensor(
        [0.9, 0.8],
    dtype=torch.float32, requires_grad=True) # 偏置形状是 (out_features,)

output = F.linear(input, weight.t(), bias)
print(output)


input_tensor = Tensor.from_numpy(input.detach().numpy(), "gpu")
weight_tensor = Tensor.from_numpy(weight.detach().numpy(), "gpu")
bias_tensor = Tensor.from_numpy(bias.detach().numpy(), "gpu")

print(input_tensor.shape())
print(weight_tensor.shape())
print(bias_tensor.shape())

y_tensor = py_tensor.fc_forward(input_tensor, weight_tensor, bias_tensor)
y_tensor.print()


# vec = Tensor.from_numpy(np.random.randn(10).astype(np.float32),"gpu")
# vec.print()
# reluvec = py_tensor.relu_forward(vec)
# reluvec.print()