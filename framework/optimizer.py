import numpy as np
from tensor import TensorFull
from operators import Tensor
import operators as ops
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build"))
import py_tensor

class Parameter(TensorFull):
    """
    标记类，用于指示该 Tensor 是一个可学习的参数。
    """
    pass

class Module:
    def __init__(self):
        self.training = True

    def parameters(self):
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Parameter):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        self.training = True
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                attr.train()
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        item.train()

    def eval(self):
        self.training = False
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                attr.eval()
            elif isinstance(attr, (list, tuple)):
                for item in attr:
                    if isinstance(item, Module):
                        item.eval()

class Optimizer:
    def __init__(self, params, lr=0.01):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            p.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.u = {}  # 动量缓存

    def step(self):
        params = []
        grads = []
        velocities = []
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            # Use cached_data directly if available to avoid unnecessary computation
            if p.cached_data is not None:
                params.append(p.cached_data)
            else:
                params.append(p.realize_cached_data())
                
            if p.grad.cached_data is not None:
                grads.append(p.grad.cached_data)
            else:
                grads.append(p.grad.realize_cached_data())
            
            if self.momentum > 0:
                if i not in self.u:
                    # Initialize velocity with zeros
                    self.u[i] = Tensor(np.zeros(p.shape, dtype=np.float32), device=p.device)
                
                if self.u[i].cached_data is not None:
                    velocities.append(self.u[i].cached_data)
                else:
                    velocities.append(self.u[i].realize_cached_data())
        
        if not params:
            return

        py_tensor.sgd_step(params, grads, velocities, self.lr, self.momentum, self.weight_decay)

class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self):
        self.t += 1
        params = []
        grads = []
        ms = []
        vs = []
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            # Use cached_data directly if available
            if p.cached_data is not None:
                params.append(p.cached_data)
            else:
                params.append(p.realize_cached_data())
                
            if p.grad.cached_data is not None:
                grads.append(p.grad.cached_data)
            else:
                grads.append(p.grad.realize_cached_data())
            
            if i not in self.m:
                self.m[i] = Tensor(np.zeros(p.shape, dtype=np.float32), device=p.device)
                self.v[i] = Tensor(np.zeros(p.shape, dtype=np.float32), device=p.device)
            
            if self.m[i].cached_data is not None:
                ms.append(self.m[i].cached_data)
            else:
                ms.append(self.m[i].realize_cached_data())
                
            if self.v[i].cached_data is not None:
                vs.append(self.v[i].cached_data)
            else:
                vs.append(self.v[i].realize_cached_data())
            
        if not params:
            return

        py_tensor.adam_step(params, grads, ms, vs, 
                            self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, self.t)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化权重 (Kaiming Uniform or Xavier)
        limit = np.sqrt(6 / (in_features + out_features))
        weight_data = np.random.uniform(-limit, limit, (in_features, out_features)).astype(dtype)
        self.weight = Parameter(weight_data, device=device, dtype=dtype)
        
        if bias:
            bias_data = np.zeros((out_features,), dtype=dtype)
            self.bias = Parameter(bias_data, device=device, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x):
        # x: (N, in_features)
        # weight: (in_features, out_features)
        # bias: (out_features,)
        if self.bias:
            return x.fc(self.weight, self.bias)
        else:
            return x @ self.weight

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if kernel_size != 3:
            raise ValueError("Only kernel_size=3 is supported")
        if stride != 1:
            raise ValueError("Only stride=1 is supported")
        if padding != 1:
            raise ValueError("Only padding=1 is supported")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化权重 (Kaiming He)
        limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size))
        weight_data = np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size)).astype(dtype)
        self.weight = Parameter(weight_data, device=device, dtype=dtype)
        
        if bias:
            bias_data = np.zeros((out_channels,), dtype=dtype)
            self.bias = Parameter(bias_data, device=device, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x):
        out = x.conv2d(self.weight)
        if self.bias:
            return out + self.bias.reshape((1, self.out_channels, 1, 1)).broadcast_to(out.shape)
        return out

class ConvReLU(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if kernel_size != 3:
            raise ValueError("Only kernel_size=3 is supported")
        if stride != 1:
            raise ValueError("Only stride=1 is supported")
        if padding != 1:
            raise ValueError("Only padding=1 is supported")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化权重 (Kaiming He)
        limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels * kernel_size * kernel_size))
        weight_data = np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size)).astype(dtype)
        self.weight = Parameter(weight_data, device=device, dtype=dtype)

        if bias:
            bias_data = np.zeros((out_channels,), dtype=dtype)
            self.bias = Parameter(bias_data, device=device, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x):
        if self.bias:
            return x.conv2d_relu(self.weight, bias=self.bias, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        return x.conv2d_relu(self.weight, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        
    def forward(self, x):
        return x.max_pool2d()

class Flatten(Module):
    def forward(self, x):
        # x: (N, C, H, W) -> (N, C*H*W)
        batch_size = x.shape[0]
        return x.reshape((batch_size, -1))

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules
        
    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x
    
    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params
    
    def train(self):
        for module in self.modules:
            module.train()
            
    def eval(self):
        for module in self.modules:
            module.eval()

class CrossEntropyLoss(Module):
    def forward(self, input, target):
        return input.cross_entropy(target)


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, device=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(np.ones((num_features,), dtype=np.float32), device=device)
        self.bias = Parameter(np.zeros((num_features,), dtype=np.float32), device=device)
        self.running_mean = Tensor(np.zeros((num_features,), dtype=np.float32), device=device, requires_grad=False)
        self.running_var = Tensor(np.ones((num_features,), dtype=np.float32), device=device, requires_grad=False)

    def forward(self, x):
        return ops.batch_norm2d(x, self.weight, self.bias, self.running_mean, self.running_var, 
                              self.momentum, self.eps, self.training)


class Dropout(Module):
    def __init__(self, p=0.5, device=None):
        super().__init__()
        self.p = p

    def forward(self, x):
        return ops.dropout(x, self.p, self.training)

