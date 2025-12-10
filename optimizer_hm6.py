import numpy as np
from tensor_hm6 import TensorFull
from operators_hm6 import Tensor

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
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad.data
            if self.weight_decay > 0:
                grad = grad + p.data * self.weight_decay

            if self.momentum > 0:
                if i not in self.u:
                    self.u[i] = grad
                else:
                    self.u[i] = self.u[i] * self.momentum + grad
                
                # 更新参数: p = p - lr * u
                p.data = p.data - self.u[i] * self.lr
            else:
                # 更新参数: p = p - lr * grad
                p.data = p.data - grad * self.lr

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
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            
            grad = p.grad.data
            if self.weight_decay > 0:
                grad = grad + p.data * self.weight_decay

            if i not in self.m:
                self.m[i] = 0 # 这里的0会在计算时广播或被替换
                self.v[i] = 0

            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
            
            # 注意：这里的运算都是基于 Tensor 的重载运算符
            if self.m[i] == 0:
                 self.m[i] = grad * (1 - self.beta1)
            else:
                 self.m[i] = self.m[i] * self.beta1 + grad * (1 - self.beta1)
            
            if self.v[i] == 0:
                 self.v[i] = (grad * grad) * (1 - self.beta2)
            else:
                 self.v[i] = self.v[i] * self.beta2 + (grad * grad) * (1 - self.beta2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # p = p - lr * m_hat / (sqrt(v_hat) + eps)
            # 由于 Tensor 可能没有 sqrt 方法，我们可能需要 pow(0.5)
            p.data = p.data - m_hat * self.lr / (v_hat ** 0.5 + self.eps)

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
        # x: (N, C_in, H, W)
        # weight: (C_out, C_in, K, K)
        # 注意：目前的底层 conv2d_forward 不支持 stride 和 padding 参数，
        # 且固定实现了 padding=1 (Same padding) 和 stride=1
        
        out = x.conv2d(self.weight)
        
        if self.bias:
            # bias: (C_out,) -> (1, C_out, 1, 1) -> broadcast
            # 需要 reshape 和 broadcast
            # 假设 out shape 是 (N, C_out, H_out, W_out)
            # 我们需要手动 reshape bias
            # 由于 Tensor 没有 view/reshape 操作符重载（有 reshape 方法），我们需要计算目标 shape
            
            # 暂时无法动态获取 out 的 shape (它是 Tensor)，除非 realize_cached_data
            # 但构建图时可能不需要数据。
            # 不过我们的 Tensor 是动态图，realize_cached_data 会执行计算。
            
            # 构造 bias 的 reshape
            b = self.bias.reshape((1, self.out_channels, 1, 1))
            out = out + b.broadcast_to(out.shape)
            
        return out

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if kernel_size != 2:
            raise ValueError("Only kernel_size=2 is supported")
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        # 同样，底层 max_pool2d_forward 似乎固定了 stride=2, kernel=2?
        # 查看 pybind_tensor.cpp: out_h = in_h / 2; out_w = in_w / 2;
        # 是的，底层写死了 2x2 pooling。
        return x.max_pool2d()

class ReLU(Module):
    def forward(self, x):
        # 使用 operators_hm6 中的 relu 函数
        from operators_hm6 import relu
        return relu(x)

class Flatten(Module):
    def forward(self, x):
        # x: (N, C, H, W) -> (N, C*H*W)
        # 获取 batch size
        # 注意：x.shape 返回的是 tuple
        batch_size = x.shape[0]
        return x.reshape((batch_size, -1))
    
class CrossEntropyLoss(Module):
    def forward(self, input, target):
        # input: (N, C)
        # target: (N,) indices or (N, C) one-hot?
        # Based on pybind_tensor.cpp comments, it seems to expect (N,) indices.
        # But let's check if we need to convert one-hot to indices or vice versa.
        # For now, assume target is (N,) indices (as floats).
        from operators_hm6 import cross_entropy
        return cross_entropy(input, target)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules_list = modules
        # 将 modules 注册为属性以便 parameters() 扫描到
        for i, module in enumerate(modules):
            setattr(self, str(i), module)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x

