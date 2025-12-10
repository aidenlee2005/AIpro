"""
本文件我们给出一个基本完善的Tensor类
你可以将hw5的对应代码复制到这里
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from device import cpu, Device
from basic_operator import Op, Value
from autodiff_hm6 import compute_gradient_of_variables

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
        return np.array(numpy_array, dtype=dtype)

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
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        return cpu()


    def backward(self, out_grad=None):
        raise NotImplementedError()
        

    def __repr__(self):
        return "Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()

        return data


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

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__

class TensorOp(Op):
    def __call__(self, *args):
        tensor_cls = type(args[0])
        return tensor_cls.make_from_op(self, args)


class EWiseAdd(TensorOp):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: np.ndarray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: np.ndarray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """逐点乘方，用标量做指数"""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: np.ndarray) -> np.ndarray:
        return a ** self.scalar
        

    def gradient(self, out_grad, node):
        return out_grad * self.scalar * (node.inputs[0] ** (self.scalar - 1))
        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """逐点乘方"""

    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a**b

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

    def compute(self, a, b):
        if np.any(abs(b) < 1e-9):
            raise ValueError("Division by zero is not allowed.")
        else:
            return a / b
        

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

    def compute(self, a):
        if self.scalar == 0:
            raise ValueError("Division by zero is not allowed.")
        return a / self.scalar
        

    def gradient(self, out_grad, node):
        if self.scalar == 0:
            raise ValueError("Division by zero is not allowed.")
        return out_grad / self.scalar
        


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ndim = len(a.shape)

        # Case 1: axes=None → swap last two dimensions
        if self.axes is None:
            axes = list(range(ndim))
            axes[-2], axes[-1] = axes[-1], axes[-2]
            axes = tuple(axes)

        # Case 2: axes=(i,j) → swap these two dims
        elif len(self.axes) == 2:
            i, j = self.axes
            i %= ndim
            j %= ndim
            axes = list(range(ndim))
            axes[i], axes[j] = axes[j], axes[i]
            axes = tuple(axes)

        # Case 3: full permutation
        else:
            axes = tuple(ax % ndim for ax in self.axes)
            if sorted(axes) != list(range(ndim)):
                raise ValueError("axes must be a permutation of all dims")

        return np.transpose(a, axes=axes)

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

    def compute(self, a):
        return np.reshape(a, self.shape)
        raise NotImplementedError()
        
    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        original_shape = node.inputs[0].shape
        return reshape(out_grad, original_shape)
        raise NotImplementedError()
        


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ## 请于此填写你的代码
        return np.broadcast_to(a, self.shape)
        raise NotImplementedError()
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
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
        grad = reshape(grad, x_shape)
        return grad
        


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return np.sum(a, axis=self.axes)
        

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        x_shape = x.shape
        if self.axes is None:
            shape = (1,) * len(x_shape)
        else:
            shape = list(x_shape)
            for axis in self.axes:
                shape[axis] = 1
            shape = tuple(shape)    
        grad_reshaped = reshape(out_grad, shape)
        return  broadcast_to(grad_reshaped, x_shape)  


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

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
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad

def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        if np.any(a <= 0):
            raise ValueError("Logarithm of non-positive values is not allowed.")
        return np.log(a)
        raise NotImplementedError()
        

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]
        raise NotImplementedError()
        


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ## 请于此填写你的代码
        return np.exp(a)
        raise NotImplementedError()
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return out_grad * exp(node.inputs[0])
        raise NotImplementedError()
        


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return np.maximum(0, a)
        raise NotImplementedError()
        

    def gradient(self, out_grad, node):
        ## 请于此填写你的代码
        return out_grad * (node.inputs[0] > 0)
        raise NotImplementedError()
    

def relu(a):
    return ReLU()(a)
