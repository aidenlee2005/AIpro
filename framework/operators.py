"""
本文件我们给出一个基本完善的Tensor类
你可以将hw5的对应代码复制到这里
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from device import cpu, Device
from basic_operator import Op, Value
from autodiff import compute_gradient_of_variables

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build"))

import py_tensor as py
from py_tensor import Tensor as MyTensor

class Tensor(Value):
    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if device is None:
            device_str = "cpu"
        elif isinstance(device, str):
            device_str = device
        else:
            # Handle Device objects
            d_str = str(device).lower()
            if "gpu" in d_str or "cuda" in d_str:
                device_str = "gpu"
            else:
                device_str = "cpu"
        return MyTensor.from_numpy(numpy_array, device=device_str)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not tensor.requires_grad:
            return tensor.detach()
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        if isinstance(data, np.ndarray):
            data = MyTensor.from_numpy(data.astype(np.float32))
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape()

    @property
    def dtype(self):
        return np.float32

    @property
    def device(self):
        data = self.realize_cached_data()
        if data.is_gpu():
            return "gpu"
        return cpu()


    def backward(self, out_grad=None):
        raise NotImplementedError()
        

    def __repr__(self):
        return "Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        return data.to_numpy()


    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return EWisePow()(self, other)
        else:
            return PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        else:
            return DivScalar(other)(self)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def matmul(self, other):
        return MatMul()(self, other)

    def sum(self, axes=None):
        return Summation(axes)(self)

    def broadcast_to(self, shape):
        return BroadcastTo(shape)(self)

    def reshape(self, shape):
        return Reshape(shape)(self)

    def __neg__(self):
        return Negate()(self)

    def transpose(self, axes=None):
        return Transpose(axes)(self)

    def conv2d(self, weight):
        return Conv2D()(self, weight)

    def max_pool2d(self):
        return MaxPool2D()(self)

    def fc(self, weight, bias):
        return FC()(self, weight, bias)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__


class TensorOp(Op):
    def __call__(self, *args):
        tensor_cls = type(args[0])
        return tensor_cls.make_from_op(self, args)


class EWiseAdd(TensorOp):
    def compute(self, a: MyTensor, b: MyTensor):
        if a.size() == 1:
            return py.scalar_add(b, a.to_numpy().item())
        if b.size() == 1:
            return py.scalar_add(a, b.to_numpy().item())
        # C++ now supports broadcasting
        return py.eltwise_add(a, b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: MyTensor):
        return py.scalar_add(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: MyTensor, b: MyTensor):
        if a.size() == 1:
            return py.scalar_mul(b, a.to_numpy().item())
        if b.size() == 1:
            return py.scalar_mul(a, b.to_numpy().item())
        # C++ now supports broadcasting
        return py.eltwise_mul(a, b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: MyTensor):
        return py.scalar_mul(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """逐点乘方，用标量做指数"""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: MyTensor) -> MyTensor:
        return py.scalar_pow(a, self.scalar)
        

    def gradient(self, out_grad, node):
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1))
        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """逐点乘方"""

    def compute(self, a: MyTensor, b: MyTensor) -> MyTensor:
        if b.size() == 1:
            return py.scalar_pow(a, b.to_numpy().item())
        # C++ now supports broadcasting
        return py.eltwise_pow(a, b)

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """逐点相除"""

    def compute(self, a: MyTensor, b: MyTensor):
        if b.size() == 1:
            return py.scalar_div(a, b.to_numpy().item())
        # C++ now supports broadcasting
        return py.eltwise_div(a, b)
        

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")
        return out_grad / node.inputs[1], -out_grad * node.inputs[0] / (node.inputs[1] ** 2)
        

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: MyTensor):
        return py.scalar_div(a, self.scalar)
        

    def gradient(self, out_grad, node):
        return out_grad / self.scalar
        


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: MyTensor):
        return MyTensor.from_numpy(np.transpose(a.to_numpy(), self.axes), device="gpu")

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        ndim = len(x.shape)

        # Recompute axes (same logic as forward)
        if self.axes is None:
            axes = list(range(ndim))
            axes[-2], axes[-1] = axes[-1], axes[-2]
            axes = tuple(axes)
        elif len(self.axes) == 2:
            i, j = self.axes
            i %= ndim
            j %= ndim
            axes = list(range(ndim))
            axes[i], axes[j] = axes[j], axes[i]
            axes = tuple(axes)
        else:
            axes = tuple(ax % ndim for ax in self.axes)

        # Compute inverse permutation
        inverse = [0] * ndim
        for out_pos, in_pos in enumerate(axes):
            inverse[in_pos] = out_pos

        return transpose(out_grad, axes=tuple(inverse))


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: MyTensor):
        return MyTensor.from_numpy(a.to_numpy().reshape(self.shape), device="gpu")
        
    def gradient(self, out_grad, node):
        return reshape(out_grad, node.inputs[0].shape)
        


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: MyTensor):
        return MyTensor.from_numpy(np.broadcast_to(a.to_numpy(), self.shape), device="gpu")
        

    def gradient(self, out_grad, node):
        x_shape = tuple(node.inputs[0].shape)
        out_shape = tuple(self.shape)
        ndim_x = len(x_shape)
        ndim_out = len(out_shape)
        
        if (ndim_out > ndim_x):
            in_shape = (1,) * (ndim_out - ndim_x) + x_shape
        else:
            in_shape = x_shape
        
        axes = tuple(i for i, (in_d, out_d) in enumerate(zip(in_shape, out_shape)) if in_d == 1 and out_d != 1)
        
        if axes:
            grad = summation(out_grad, axes=axes)
        else:
            grad = out_grad
        return reshape(grad, x_shape)
        


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: MyTensor):
        return MyTensor.from_numpy(np.sum(a.to_numpy(), axis=self.axes), device="gpu")
        

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        x_shape = x.shape
        if self.axes is None:
            shape = (1,) * len(x_shape)
        else:
            shape = list(x_shape)
            if isinstance(self.axes, int):
                shape[self.axes] = 1
            else:
                for axis in self.axes:
                    shape[axis] = 1
            shape = tuple(shape)    
        grad_reshaped = reshape(out_grad, shape)
        return  broadcast_to(grad_reshaped, x_shape)  


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: MyTensor, b: MyTensor):
        return MyTensor.from_numpy(a.to_numpy() @ b.to_numpy(), device="gpu")

    def gradient(self, out_grad, node):
        a, b = node.inputs
        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)
        
        def reduce_broadcast(grad, target_shape):
            if grad.shape == target_shape:
                return grad
            ndim_diff = len(grad.shape) - len(target_shape)
            if ndim_diff > 0:
                grad = grad.sum(axes=tuple(range(ndim_diff)))
            axes_to_sum = []
            for i, dim in enumerate(target_shape):
                if dim == 1 and grad.shape[i] > 1:
                    axes_to_sum.append(i)
            
            if axes_to_sum:
                grad = grad.sum(axes=tuple(axes_to_sum))

            return grad.reshape(target_shape)

        grad_a = reduce_broadcast(grad_a, a.shape)
        grad_b = reduce_broadcast(grad_b, b.shape)
        return grad_a, grad_b



def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: MyTensor):
        return py.scalar_mul(a, -1.0)

    def gradient(self, out_grad, node):
        return -out_grad

def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: MyTensor):
        return MyTensor.from_numpy(np.log(a.to_numpy()), device="gpu")
        

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]
        


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: MyTensor):
        return MyTensor.from_numpy(np.exp(a.to_numpy()), device="gpu")
        

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])
        


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: MyTensor):
        return py.relu_forward(a)
        

    def gradient(self, out_grad: Tensor, node: Tensor):
        return Tensor.make_const(py.relu_backward(out_grad.realize_cached_data(), node.realize_cached_data()))
    

def relu(a):
    return ReLU()(a)


class Conv2D(TensorOp):
    def compute(self, input: MyTensor, weight: MyTensor):
        return py.conv2d_forward(input, weight)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input, weight = node.inputs[0], node.inputs[1]
        # Initialize gradients with zeros (using numpy for safety as py_tensor might not zero-init)
        # Assuming device is same as input
        dev = "gpu" if input.realize_cached_data().is_gpu() else "cpu"
        
        # We need shapes to create zero tensors. 
        # input and weight are Values, realize_cached_data() gives MyTensor
        input_t = input.realize_cached_data()
        weight_t = weight.realize_cached_data()
        out_grad_t = out_grad.realize_cached_data()
        
        grad_input_t = MyTensor.from_numpy(np.zeros(input_t.shape(), dtype=np.float32), device=dev)
        grad_weight_t = MyTensor.from_numpy(np.zeros(weight_t.shape(), dtype=np.float32), device=dev)
        
        py.conv2d_backward(out_grad_t, input_t, weight_t, grad_input_t, grad_weight_t)
        
        return Tensor.make_const(grad_input_t), Tensor.make_const(grad_weight_t)


def conv2d(input, weight):
    return Conv2D()(input, weight)


class MaxPool2D(TensorOp):
    def compute(self, input: MyTensor):
        return py.max_pool2d_forward(input)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input = node.inputs[0]
        input_t = input.realize_cached_data()
        out_grad_t = out_grad.realize_cached_data()
        dev = "gpu" if input_t.is_gpu() else "cpu"
        
        # Recompute mask because forward didn't save it (limitation of current binding)
        mask_t = py.max_pool2d_forward_mask(input_t)
        
        grad_input_t = MyTensor.from_numpy(np.zeros(input_t.shape(), dtype=np.float32), device=dev)
        
        py.max_pool2d_backward(out_grad_t, mask_t, input_t, grad_input_t)
        
        return Tensor.make_const(grad_input_t)


def max_pool2d(input):
    return MaxPool2D()(input)


class FC(TensorOp):
    def compute(self, input: MyTensor, weight: MyTensor, bias: MyTensor):
        return py.fc_forward(input, weight, bias)

    def gradient(self, out_grad: Tensor, node: Tensor):
        input, weight, bias = node.inputs
        input_t = input.realize_cached_data()
        weight_t = weight.realize_cached_data()
        bias_t = bias.realize_cached_data()
        out_grad_t = out_grad.realize_cached_data()
        dev = "gpu" if input_t.is_gpu() else "cpu"
        
        grad_input_t = MyTensor.from_numpy(np.zeros(input_t.shape(), dtype=np.float32), device=dev)
        grad_weight_t = MyTensor.from_numpy(np.zeros(weight_t.shape(), dtype=np.float32), device=dev)
        grad_bias_t = MyTensor.from_numpy(np.zeros(bias_t.shape(), dtype=np.float32), device=dev)
        
        py.fc_backward(out_grad_t, input_t, weight_t, bias_t, grad_input_t, grad_weight_t, grad_bias_t)
        
        return Tensor.make_const(grad_input_t), Tensor.make_const(grad_weight_t), Tensor.make_const(grad_bias_t)


def fc(input, weight, bias):
    return FC()(input, weight, bias)


class CrossEntropy(TensorOp):
    def compute(self, input: MyTensor, labels: MyTensor):
        # input: (N, C) logits
        # labels: (N, C) one-hot
        # Backend returns float
        loss_val = py.cross_entropy_forward(input, labels)
        # Wrap in 1-element tensor
        return MyTensor.from_numpy(np.array(loss_val, dtype=np.float32), device="gpu" if input.is_gpu() else "cpu")

    def gradient(self, out_grad: Tensor, node: Tensor):
        input, labels = node.inputs
        input_t = input.realize_cached_data()
        labels_t = labels.realize_cached_data()
        dev = "gpu" if input_t.is_gpu() else "cpu"
        
        grad_input_t = MyTensor.from_numpy(np.zeros(input_t.shape(), dtype=np.float32), device=dev)
        
        # Backend assumes grad_output is 1.0 implicitly? 
        # Or does it just compute dLoss/dInput?
        # Usually dLoss/dInput = softmax(input) - labels (for one-hot) / N
        # Let's assume backend computes the full gradient.
        # If out_grad is not 1, we should multiply by it?
        # But usually Loss is the final scalar.
        
        py.cross_entropy_backward(input_t, labels_t, grad_input_t)
        
        # If out_grad is not 1 (e.g. part of a larger graph), we should multiply.
        # But CrossEntropy is usually the end.
        # Let's multiply by out_grad just in case, if shapes allow.
        # out_grad is scalar (1,). grad_input is (N, C).
        
        return Tensor.make_const(grad_input_t) * out_grad, None # No grad for labels

def cross_entropy(input, labels):
    return CrossEntropy()(input, labels)

