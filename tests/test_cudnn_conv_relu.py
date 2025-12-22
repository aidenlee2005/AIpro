import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../build"))
import py_tensor
import torch
import numpy as np
import unittest

class TestCudnnConvRelu(unittest.TestCase):
    def test_conv2d_relu(self):
        batch_size = 2
        in_channels = 3
        out_channels = 4
        height = 8
        width = 8
        kernel_size = 3
        stride = 1
        padding = 1

        # Inputs
        input_np = np.random.randn(batch_size, in_channels, height, width).astype(np.float32)
        filter_np = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)

        # PyTorch
        input_torch = torch.tensor(input_np, device='cuda', requires_grad=True)
        filter_torch = torch.tensor(filter_np, device='cuda', requires_grad=True)
        
        # Conv2d + ReLU
        output_torch = torch.nn.functional.conv2d(input_torch, filter_torch, padding=padding, stride=stride)
        output_torch = torch.nn.functional.relu(output_torch)
        
        # Backward
        grad_output_np = np.random.randn(*output_torch.shape).astype(np.float32)
        grad_output_torch = torch.tensor(grad_output_np).cuda()
        output_torch.backward(grad_output_torch)
        
        grad_input_torch = input_torch.grad.cpu().numpy()
        grad_filter_torch = filter_torch.grad.cpu().numpy()
        output_torch_np = output_torch.detach().cpu().numpy()

        # My Framework
        input_my = py_tensor.Tensor.from_numpy(input_np, "gpu")
        filter_my = py_tensor.Tensor.from_numpy(filter_np, "gpu")
        
        # Forward
        output_my = py_tensor.conv2d_relu_forward(input_my, filter_my, kernel_size, stride, padding)
        output_my_np = np.array(output_my.to_numpy())
        
        # Backward
        grad_output_my = py_tensor.Tensor.from_numpy(grad_output_np, "gpu")
        grad_input_my, grad_filter_my = py_tensor.conv2d_relu_backward(grad_output_my, output_my, input_my, filter_my, kernel_size, stride, padding)
        
        grad_input_my_np = np.array(grad_input_my.to_numpy())
        grad_filter_my_np = np.array(grad_filter_my.to_numpy())

        # Compare
        print("Comparing Output...")
        np.testing.assert_allclose(output_my_np, output_torch_np, rtol=1e-4, atol=1e-4)
        print("Output OK")
        
        print("Comparing Grad Input...")
        np.testing.assert_allclose(grad_input_my_np, grad_input_torch, rtol=1e-4, atol=1e-4)
        print("Grad Input OK")
        
        print("Comparing Grad Filter...")
        np.testing.assert_allclose(grad_filter_my_np, grad_filter_torch, rtol=1e-4, atol=1e-4)
        print("Grad Filter OK")

if __name__ == '__main__':
    unittest.main()
