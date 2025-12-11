
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build"))
import py_tensor
import numpy as np
import torch
import torch.nn as nn

def test_batchnorm_grad():
    print("Testing BatchNorm Gradients...")
    
    batch_size = 4
    channels = 3
    height = 4
    width = 4
    
    # Setup inputs
    np.random.seed(42)
    x_np = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    w_np = np.random.randn(channels).astype(np.float32)
    b_np = np.random.randn(channels).astype(np.float32)
    grad_out_np = np.random.randn(batch_size, channels, height, width).astype(np.float32)
    
    # PyTorch Reference
    x_torch = torch.tensor(x_np, requires_grad=True)
    w_torch = torch.tensor(w_np, requires_grad=True)
    b_torch = torch.tensor(b_np, requires_grad=True)
    
    bn_torch = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
    bn_torch.weight.data = w_torch
    bn_torch.bias.data = b_torch
    bn_torch.running_mean.zero_()
    bn_torch.running_var.fill_(1.0)
    
    out_torch = bn_torch(x_torch)
    out_torch.backward(torch.tensor(grad_out_np))
    
    grad_x_torch = x_torch.grad.numpy()
    grad_w_torch = bn_torch.weight.grad.numpy()
    grad_b_torch = bn_torch.bias.grad.numpy()
    
    # Custom Implementation
    x_custom = py_tensor.Tensor.from_numpy(x_np, "gpu")
    w_custom = py_tensor.Tensor.from_numpy(w_np, "gpu")
    b_custom = py_tensor.Tensor.from_numpy(b_np, "gpu")
    running_mean_custom = py_tensor.Tensor.from_numpy(np.zeros(channels, dtype=np.float32), "gpu")
    running_var_custom = py_tensor.Tensor.from_numpy(np.ones(channels, dtype=np.float32), "gpu")
    
    # Forward
    out_custom, save_mean, save_inv_std = py_tensor.batch_norm_forward_training(
        x_custom, w_custom, b_custom, running_mean_custom, running_var_custom, 0.1, 1e-5
    )
    
    # Backward
    grad_out_custom = py_tensor.Tensor.from_numpy(grad_out_np, "gpu")
    grad_x_custom, grad_w_custom, grad_b_custom = py_tensor.batch_norm_backward(
        grad_out_custom, x_custom, w_custom, save_mean, save_inv_std
    )
    
    # Compare
    print("Checking Gradients...")
    
    diff_x = np.abs(grad_x_custom.to_numpy() - grad_x_torch).max()
    diff_w = np.abs(grad_w_custom.to_numpy() - grad_w_torch).max()
    diff_b = np.abs(grad_b_custom.to_numpy() - grad_b_torch).max()
    
    print(f"Max Diff Input Grad: {diff_x}")
    print(f"Max Diff Weight Grad: {diff_w}")
    print(f"Max Diff Bias Grad: {diff_b}")
    
    if diff_x < 1e-4 and diff_w < 1e-4 and diff_b < 1e-4:
        print("SUCCESS: Gradients match PyTorch!")
    else:
        print("FAILURE: Gradients do not match!")

    # Check Running Stats
    print("\nChecking Running Stats...")
    rm_torch = bn_torch.running_mean.numpy()
    rv_torch = bn_torch.running_var.numpy()
    
    rm_custom = running_mean_custom.to_numpy()
    rv_custom = running_var_custom.to_numpy()
    
    diff_rm = np.abs(rm_custom - rm_torch).max()
    diff_rv = np.abs(rv_custom - rv_torch).max()
    
    print(f"Max Diff Running Mean: {diff_rm}")
    print(f"Max Diff Running Var: {diff_rv}")
    
    if diff_rm < 1e-5 and diff_rv < 1e-5:
        print("SUCCESS: Running stats match PyTorch!")
    else:
        print("FAILURE: Running stats do not match!")

if __name__ == "__main__":
    test_batchnorm_grad()
