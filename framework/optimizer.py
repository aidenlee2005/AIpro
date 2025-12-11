import numpy as np
from tensor import TensorFull
from operators import Tensor
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build"))
import py_tensor

class Parameter(TensorFull):
    """
    标记类，用于指示该 Tensor 是一个可学习的参数。
    """
    pass

class Module:
    """
    所有神经网络模块的基类。
    """
    def __init__(self):
        self.training = True

    def parameters(self):
        """
        返回该模块及其子模块中的所有 Parameter 对象。
        """
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

    def eval(self):
        self.training = False
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Module):
                attr.eval()

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
            
            params.append(p.realize_cached_data())
            grads.append(p.grad.realize_cached_data())
            
            if self.momentum > 0:
                if i not in self.u:
                    # Initialize velocity with zeros
                    self.u[i] = Tensor(np.zeros(p.shape, dtype=np.float32), device=p.device)
                
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
            
            params.append(p.realize_cached_data())
            grads.append(p.grad.realize_cached_data())
            
            if i not in self.m:
                self.m[i] = Tensor(np.zeros(p.shape, dtype=np.float32), device=p.device)
                self.v[i] = Tensor(np.zeros(p.shape, dtype=np.float32), device=p.device)
            
            ms.append(self.m[i].realize_cached_data())
            vs.append(self.v[i].realize_cached_data())
            
        if not params:
            return

        py_tensor.adam_step(params, grads, ms, vs, 
                            self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, self.t)

# --- 常用层定义 ---

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
        # 使用 TensorFull 的 fc 方法 (底层调用 py.fc_forward)
        if self.bias:
            return x.fc(self.weight, self.bias)
        else:
            # 如果没有 bias，构造一个全0 bias 或者使用 matmul
            # 这里为了兼容 fc 接口，构造一个临时的 0 bias
            # 注意：这可能会带来额外的开销，实际中应优化
            # 假设 x 在 GPU，bias 也应在 GPU
            # 简单起见，这里假设必须有 bias，或者回退到 matmul
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

class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype="float32"):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if affine:
            self.weight = Parameter(np.ones((num_features,), dtype=dtype), device=device, dtype=dtype)
            self.bias = Parameter(np.zeros((num_features,), dtype=dtype), device=device, dtype=dtype)
        else:
            self.weight = None
            self.bias = None
            
        if track_running_stats:
            self.running_mean = Tensor(np.zeros((num_features,), dtype=dtype), device=device, requires_grad=False)
            self.running_var = Tensor(np.ones((num_features,), dtype=dtype), device=device, requires_grad=False)
        else:
            self.running_mean = None
            self.running_var = None

    def forward(self, x):
        import operators as F
        
        w = self.weight if self.weight else Tensor(np.ones((self.num_features,), dtype="float32"), device=x.device, requires_grad=False)
        b = self.bias if self.bias else Tensor(np.zeros((self.num_features,), dtype="float32"), device=x.device, requires_grad=False)
        rm = self.running_mean if self.running_mean else Tensor(np.zeros((self.num_features,), dtype="float32"), device=x.device, requires_grad=False)
        rv = self.running_var if self.running_var else Tensor(np.ones((self.num_features,), dtype="float32"), device=x.device, requires_grad=False)
        
        return F.batch_norm2d(x, w, b, rm, rv, self.momentum, self.eps, self.training)

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride else kernel_size
        self.padding = padding
        
    def forward(self, x):
        # Currently only supports 2x2 max pool
        return x.max_pool2d()

class Flatten(Module):
    def forward(self, x):
        # x: (N, C, H, W) -> (N, C*H*W)
        # We need reshape.
        # If Tensor doesn't have reshape, we can use numpy reshape on data?
        # No, that breaks graph.
        # But `operators.py` likely has `reshape` or `flatten`.
        # Let's check `operators.py` for `reshape`.
        # If not, we can implement it.
        # But `examples/train_cifar10_custom.py` uses `nn.Flatten`.
        # So `Flatten` must be implemented or `Tensor` has `reshape`.
        
        # Assuming Tensor has reshape.
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

