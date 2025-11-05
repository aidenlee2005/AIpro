### 关于接口说明

**FC Layer**

```python
fc_forward(input:Tensor, weight:Tensor, bias:Tensor) -> output:Tensor
# input: (batch_size, input_size)
# weight: (output_size, input_size)
# bias: (output_size)
# output: (batch_size, output_size)

fc_backward(grad_output:Tensor, input:Tensor, weight:Tensor, bias:Tensor, grad_input:Tensor, grad_weight:Tensor, grad_bias:Tensor) -> None
# grad_output: (batch_size, output_size)
# input: (batch_size, input_size)
# weight: (output_size, input_size)
# bias: (output_size)
# grad_input: (batch_size, input_size)
# grad_weight: (output_size, input_size)
# grad_bias: (output_size)
```

**Conv2d Layer**

```python
conv2d_forward(input:Tensor, filter:Tensor) -> output:Tensor
# input: (batch_size, input_channels, iH, iW)
# filter: (output_channels, input_channels, 3, 3)
# output: (batch_size, output_channels, iH, iW)

conv2d_backward(grad_output:Tensor, input:Tensor, filter:Tensor, grad_input:Tensor, grad_weight:Tensor) -> None
# grad_output: (batch_size, output_channels, iH, iW)
# input: (batch_size, input_channels, iH, iW)
# filter: (output_channels, input_channels, 3, 3)
# grad_input: (batch_size, input_channels, iH, iW)
# grad_weight: (output_channels, input_channels, 3, 3)
```

**Max Pooling Layer**

```python
max_pool2d_forward(input:Tensor) -> output:Tensor
# input: (batch_size, input_channels, iH, iW)
# output: (batch_size, input_channels, oH, oW)

max_pool2d_forward_mask(input:Tensor) -> mask:Tensor
# input: (batch_size, input_channels, iH, iW)
# mask: (batch_size, input_channels, oH, oW)

max_pool2d_backward(grad_output:Tensor, mask:Tensor, input:Tensor, grad_input:Tensor) -> None
# grad_output: (batch_size, input_channels, oH, oW)
# mask: (batch_size, input_channels, oH, oW)
# input: (batch_size, input_channels, iH, iW)
# grad_input: (batch_size, input_channels, iH, iW)
```

**Sigmoid Layer**

```python
sigmoid_forward(input:Tensor) -> output:Tensor
# input: (*)
# output: (*)

sigmoid_backward(grad_output:Tensor, grad_input:Tensor) -> input:Tensor
# input: (*)
# grad_output: (*)
# grad_input: (*)
```

**Relu Layer**

```python 
relu_forward(input:Tensor) -> output:Tensor
# input: (*)

relu_backward(grad_output:Tensor, grad_input:Tensor) -> input:Tensor
# input: (*)
# grad_output: (*)
# grad_input: (*)
```

**Softmax & CrossEntropy Loss Layer**

```python
softmax_forward(input:Tensor) -> output:Tensor
# input: (batch_size, input_size)
# output: (batch_size, input_size)

cross_entropy_forward(logits:Tensor, labels:Tensor) -> loss:float
# logits: (batch_size, num_classes)
# labels: (batch_size)

cross_entropy_backward(logits:Tensor, labels:Tensor, grad_logits:Tensor) -> None
# logits: (batch_size, num_classes)
# labels: (batch_size)
# grad_logits: (batch_size, num_classes)
```

### Python中的调用

**Tensor与np.ndarray的转换**

```python
make_tensor(data:np.ndarray, dtype:str) -> Tensor

to_numpy_from_py(tensor:Tensor) -> np.ndarray
```