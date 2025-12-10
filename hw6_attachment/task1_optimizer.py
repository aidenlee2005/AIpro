"""
本文件我们尝试实现一个Optimizer类，用于优化一个简单的双层Linear Network
本次作业主要的内容将会在opti_epoch内对于一个epoch的参数进行优化
分为SGD_epoch和Adam_epoch两个函数，分别对应SGD和Adam两种优化器
其余函数为辅助函数，也请一并填写
和大作业的要求一致，我们不对数据处理和读取做任何要求
因此你可以引入任何的库来帮你进行数据处理和读取
理论上我们也不需要依赖hw5的内容，如果你需要的话，你可以將hw5对应代码copy到对应位置
"""
# from task0_autodiff import *
# from task0_operators import *
import numpy as np
import matplotlib.pyplot as plt

def parse_mnist():
    """
    读取MNIST数据集，并进行简单的处理，如归一化
    你可以可以引入任何的库来帮你进行数据处理和读取
    所以不会规定你的输入的格式
    但需要使得输出包括X_tr, y_tr和X_te, y_te
    """
    ## 请于此填写你的代码
    from torchvision import datasets
    train = datasets.MNIST('./data', train=True, download=True)
    test = datasets.MNIST('./data', train=False, download=True)
    X_tr = train.data.numpy().reshape(train.data.shape[0], -1).astype(np.float32) / 255.0
    y_tr = train.targets.numpy().astype(np.int32)
    X_te = test.data.numpy().reshape(test.data.shape[0], -1).astype(np.float32) / 255.0
    y_te = test.targets.numpy().astype(np.int32)
    return X_tr, y_tr, X_te, y_te
    raise NotImplementedError()

def set_structure(n, hidden_dim, k):
    """
    定义你的网络结构，并进行简单的初始化
    一个简单的网络结构为两个Linear层，中间加上ReLU
    Args:
        n: input dimension of the data.
        hidden_dim: hidden dimension of the network.
        k: output dimension of the network, which is the number of classes.
    Returns:
        List of Weights matrix.
    Example:
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return list(W1, W2)
    """
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return [W1, W2]
    ## 请于此填写你的代码
    
def forward(X, weights):
    """
    使用你的网络结构，来计算给定输入X的输出
    Args:
        X : 2D input array of size (num_examples, input_dim).
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
    Returns:
        Logits calculated by your network structure.
    Example:
    W1 = weights[0]
    W2 = weights[1]
    return np.maximum(X@W1,0)@W2
    """
    ## 请于此填写你的代码
    W1 = weights[0]
    W2 = weights[1]
    return np.maximum(X @ W1, 0) @ W2
    raise NotImplementedError()

def softmax_loss(Z, y):
    """ 
    一个写了很多遍的Softmax loss...

    Args:
        Z : 2D numpy array of shape (batch_size, num_classes), 
        containing the logit predictions for each class.
        y : 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ## 请于此填写你的代码
    Z_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    softmax = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
    batch_size = Z.shape[0]
    loss = -np.log(softmax[np.arange(batch_size), y] + 1e-15)
    return np.mean(loss)
    raise NotImplementedError()

def opti_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """
    优化一个epoch
    具体请参考SGD_epoch 和 Adam_epoch的代码
    """
    if using_adam:
        Adam_epoch(X, y, weights, lr = lr, batch=batch, beta1=beta1, beta2=beta2)
    else:
        SGD_epoch(X, y, weights, lr = lr, batch=batch)

def SGD_epoch(X, y, weights, lr = 0.1, batch=100):
    """ 
    SGD优化一个List of Weights
    本函数应该inplace地修改Weights矩阵来进行优化
    用学习率简单更新Weights

    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ## 请于此填写你的代码
    W1 = weights[0]
    W2 = weights[1]
    num_examples = X.shape[0]
    random_idx = np.random.permutation(num_examples)
    for i in range(0, num_examples, batch):
        batch_idx = random_idx[i:i+batch]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]

        # Forward pass
        hidden = np.maximum(X_batch @ W1, 0)
        logits = hidden @ W2

        # Compute gradients
        Z_exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
        softmax[np.arange(len(y_batch)), y_batch] -= 1
        softmax /= len(y_batch)

        grad_W2 = hidden.T @ softmax
        grad_hidden = softmax @ W2.T
        grad_hidden[hidden <= 0] = 0
        grad_W1 = X_batch.T @ grad_hidden

        # Update weights
        W1 -= lr * grad_W1
        W2 -= lr * grad_W2
    return

def Adam_epoch(X, y, weights, lr = 0.1, batch=100, beta1=0.9, beta2=0.999):
    """ 
    ADAM优化一个
    本函数应该inplace地修改Weights矩阵来进行优化
    使用Adaptive Moment Estimation来进行更新Weights
    具体步骤可以是：
    1. 增加时间步 $t$。
    2. 计算当前梯度 $g$。
    3. 更新一阶矩向量：$m = \beta_1 \cdot m + (1 - \beta_1) \cdot g$。
    4. 更新二阶矩向量：$v = \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$。
    5. 计算偏差校正后的一阶和二阶矩估计：$\hat{m} = m / (1 - \beta_1^t)$ 和 $\hat{v} = v / (1 - \beta_2^t)$。
    6. 更新参数：$\theta = \theta - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$。
    其中$\eta$表示学习率，$\beta_1$和$\beta_2$是平滑参数，
    $t$表示时间步，$\epsilon$是为了维持数值稳定性而添加的常数，如1e-8。
    
    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
        beta1 (float): smoothing parameter for first order momentum
        beta2 (float): smoothing parameter for second order momentum

    Returns:
        None
    """
    ## 请于此填写你的代码
    W1 = weights[0]
    W2 = weights[1]
    m1 = np.zeros_like(W1)
    m2 = np.zeros_like(W2)
    v1 = np.zeros_like(W1)  
    v2 = np.zeros_like(W2)
    num_examples = X.shape[0]
    random_idx = np.random.permutation(num_examples)
    t = 0
    for i in range(0, num_examples, batch):
        batch_idx = random_idx[i:i+batch]
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]

        # Forward pass
        hidden = np.maximum(X_batch @ W1, 0)
        logits = hidden @ W2

        # Compute gradients
        Z_exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        softmax = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
        softmax[np.arange(len(y_batch)), y_batch] -= 1
        softmax /= len(y_batch)

        grad_W2 = hidden.T @ softmax
        grad_hidden = softmax @ W2.T
        grad_hidden[hidden <= 0] = 0
        grad_W1 = X_batch.T @ grad_hidden
        
        t += 1
        
        m1 = beta1 * m1 + (1 - beta1) * grad_W1
        v1 = beta2 * v1 + (1 - beta2) * (grad_W1 ** 2)
        m1_hat = m1 / (1 - beta1 ** t)
        v1_hat = v1 / (1 - beta2 ** t)
        
        m2 = beta1 * m2 + (1 - beta1) * grad_W2
        v2 = beta2 * v2 + (1 - beta2) * (grad_W2 ** 2)
        m2_hat = m2 / (1 - beta1 ** t)
        v2_hat = v2 / (1 - beta2 ** t)
        
        # Update weights
        W1 -= lr * m1_hat / (np.sqrt(v1_hat) + 1e-8)
        W2 -= lr * m2_hat / (np.sqrt(v2_hat) + 1e-8)
    return


def loss_err(h,y):
    """ 
    计算给定预测结果h和真实标签y的loss和error
    """
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def plot_metrics(train_losses, test_losses, test_errs, save_path=None):
    """
    绘制 loss 曲线（train/test）和 test accuracy 曲线。
    test_errs 为错误率数组，绘制时转换为准确率，纵坐标自适应。
    """
    epochs = np.arange(len(train_losses))
    test_accs = 1.0 - np.array(test_errs)

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    # Loss 曲线
    axs[0].plot(epochs, train_losses, label='Train Loss', marker='o')
    axs[0].plot(epochs, test_losses, label='Test Loss', marker='o')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss over Epochs')
    axs[0].grid(True)
    axs[0].legend()

    # Test 准确率曲线（纵坐标自适应）
    axs[1].plot(epochs, test_accs, label='Test Accuracy', color='tab:green', marker='o')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Test Accuracy over Epochs')
    axs[1].grid(True)
    axs[1].legend()

    # 自适应纵坐标：根据数据范围加一定边距，但限制在 [0,1]
    if test_accs.size > 0:
        acc_min = float(np.min(test_accs))
        acc_max = float(np.max(test_accs))
        data_range = acc_max - acc_min
        # 边距：至少 0.02，或数据范围的 5%
        margin = max(0.02, 0.05 * data_range) if data_range > 0 else 0.02
        ymin = max(0.0, acc_min - margin)
        ymax = min(1.0, acc_max + margin)
        # 若 ymax==ymin（单点情况），给小幅区间
        if ymax <= ymin:
            ymin = max(0.0, acc_min - 0.01)
            ymax = min(1.0, acc_max + 0.01)
        axs[1].set_ylim(ymin, ymax)
    else:
        axs[1].set_ylim(0, 1.0)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show()
    plt.close(fig)


def train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """ 
    训练过程
    """
    n, k = X_tr.shape[1], y_tr.max() + 1
    weights = set_structure(n, hidden_dim, k)
    np.random.seed(0)
    
    lr_decay = 0.99
    current_lr = lr
    
    train_losses, test_losses, train_errs, test_errs = [], [], [], []
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        current_lr = current_lr * lr_decay
        opti_epoch(X_tr, y_tr, weights, lr=current_lr, batch=batch, beta1=beta1, beta2=beta2, using_adam=using_adam)
        train_loss, train_err = loss_err(forward(X_tr, weights), y_tr)
        test_loss, test_err = loss_err(forward(X_te, weights), y_te)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_errs.append(train_err)
        test_errs.append(test_err)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))

    # 绘图：loss（train/test）与 test accuracy
    plot_metrics(train_losses, test_losses, test_errs, "training_metrics"+str(using_adam)+".png")
    return weights, (train_losses, test_losses, train_errs, test_errs)



if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = parse_mnist() 
    weights = set_structure(X_tr.shape[1], 100, y_tr.max() + 1)
    ## using SGD optimizer 
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr = 0.2, batch=100, beta1=0.9, beta2=0.999, using_adam=False)
    ## using Adam optimizer
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr = 5e-4, batch=100, beta1=0.9, beta2=0.999, using_adam=True)
