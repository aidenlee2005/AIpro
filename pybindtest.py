import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))

import unittest
import numpy as np

def import_py_tensor_or_skip():
    try:
        import py_tensor
        return py_tensor
    except Exception as e:
        raise unittest.SkipTest(f"无法导入 py_tensor: {e}")

def to_numpy_from_py(t):
    if hasattr(t, "to_numpy"):
        return np.array(t.to_numpy())
    return np.array(t)

def make_tensor(py, arr, device="gpu"):
    return py.Tensor.from_numpy(arr.astype(np.float32), device)

class TestOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.py = import_py_tensor_or_skip()
        try:
            import torch
            import torch.nn.functional as F
            cls.torch = torch
            cls.F = F
        except Exception as e:
            raise unittest.SkipTest(f"缺少 torch 库以用于验证: {e}")

    def test_relu_forward_backward(self):
        py = self.py
        torch = self.torch
        F = self.F

        if not hasattr(py, "relu_forward"):
            self.skipTest("未导出 relu_forward")
        np.random.seed(0)
        x = (np.random.randn(64,3,28,28).astype(np.float32) - 0.5) * 2.0 
        px = make_tensor(py, x, "gpu")
        
        out = py.relu_forward(px)
        out_np = to_numpy_from_py(out)
        
        expect = F.relu(torch.from_numpy(x)).numpy()
        self.assertEqual(out_np.shape, expect.shape)
        self.assertTrue(np.allclose(out_np, expect, atol=1e-6))

        if not hasattr(py, "relu_backward"):
            self.skipTest("未导出 relu_backward")
        grad_out = np.random.randn(*expect.shape).astype(np.float32)
        gpy = make_tensor(py, grad_out, "gpu")
        grad_in = py.relu_backward(gpy, out)
        grad_in_np = to_numpy_from_py(grad_in)
        expected_grad = grad_out * (x > 0).astype(np.float32)
        self.assertTrue(np.allclose(grad_in_np, expected_grad, atol=1e-5))

    def test_sigmoid_forward_backward(self):
        py = self.py
        torch = self.torch
        F = self.F

        if not hasattr(py, "sigmoid_forward"):
            self.skipTest("未导出 sigmoid_forward")
        np.random.seed(1)
        x = np.random.randn(10,12).astype(np.float32)
        px = make_tensor(py, x, "gpu")
        out = py.sigmoid_forward(px)
        out_np = to_numpy_from_py(out)
        expect = torch.sigmoid(torch.from_numpy(x)).numpy()
        self.assertTrue(np.allclose(out_np, expect, atol=1e-6))

        if not hasattr(py, "sigmoid_backward"):
            self.skipTest("未导出 sigmoid_backward")
        grad_out = np.random.randn(*expect.shape).astype(np.float32)
        gpy = make_tensor(py, grad_out, "gpu")
        grad_in = py.sigmoid_backward(gpy, out)
        grad_in_np = to_numpy_from_py(grad_in)
        # expected via torch
        s = torch.sigmoid(torch.from_numpy(x)).numpy()
        expected_grad = grad_out * (s * (1.0 - s))
        self.assertTrue(np.allclose(grad_in_np, expected_grad, atol=1e-5))

    def test_fc_forward_and_backward(self):
        py = self.py
        torch = self.torch

        if not hasattr(py, "fc_forward"):
            self.skipTest("未导出 fc_forward")
        np.random.seed(2)
        xn = np.random.randn(128,512).astype(np.float32)   # (batch, in_features)
        wn = np.random.randn(512,128).astype(np.float32)   # (in_features, out_features)
        bn = np.random.randn(128).astype(np.float32)     # (out_features,)

        x = make_tensor(py, xn, "gpu")
        w = make_tensor(py, wn, "gpu")
        b = make_tensor(py, bn, "gpu")

        y = py.fc_forward(x, w, b)
        y_np = to_numpy_from_py(y)

        # expected using torch: x @ w + b
        expect = torch.from_numpy(xn).mm(torch.from_numpy(wn)).numpy() + bn.reshape((1, -1))
        self.assertEqual(y_np.shape, expect.shape)
        self.assertTrue(np.allclose(y_np, expect, atol=1e-3))

        if not hasattr(py, "fc_backward"):
            self.skipTest("未导出 fc_backward")
        grad_out = np.random.randn(*expect.shape).astype(np.float32)
        grad_out_t = make_tensor(py, grad_out, "gpu")

        grad_input = make_tensor(py, np.zeros_like(xn), "gpu")
        grad_weight = make_tensor(py, np.zeros_like(wn), "gpu")
        grad_bias = make_tensor(py, np.zeros_like(bn), "gpu")

        py.fc_backward(grad_out_t, x, w, b, grad_input, grad_weight, grad_bias)

        g_in_np = to_numpy_from_py(grad_input)
        g_w_np = to_numpy_from_py(grad_weight)
        g_b_np = to_numpy_from_py(grad_bias)

        # expected grads via torch
        expected_g_in = torch.from_numpy(grad_out).mm(torch.from_numpy(wn).t()).numpy()
        expected_g_w = torch.from_numpy(xn).t().mm(torch.from_numpy(grad_out)).numpy()
        expected_g_b = torch.from_numpy(grad_out).sum(dim=0).numpy()

        self.assertTrue(np.allclose(g_in_np, expected_g_in, atol=1e-3))
        self.assertTrue(np.allclose(g_w_np, expected_g_w, atol=1e-3))
        self.assertTrue(np.allclose(g_b_np.ravel(), expected_g_b.ravel(), atol=1e-3))

    def test_conv2d_forward(self):
        py = self.py
        F = self.F
        torch = self.torch

        if not hasattr(py, "conv2d_forward"):
            self.skipTest("未导出 conv2d_forward")
        np.random.seed(3)
        x = np.random.randn(4,3,28,28).astype(np.float32)
        filt = np.random.randn(6,3,3,3).astype(np.float32)
        px = make_tensor(py, x, "gpu")
        pf = make_tensor(py, filt, "gpu")
        out = py.conv2d_forward(px, pf)
        out_np = to_numpy_from_py(out)

        expect = F.conv2d(torch.from_numpy(x), torch.from_numpy(filt), bias=None, stride=1, padding=1).numpy()
        self.assertEqual(out_np.shape, expect.shape)
        self.assertTrue(np.allclose(out_np, expect, atol=1e-3))
    
    def test_conv2d_backward(self):
        py = self.py
        F = self.F
        torch = self.torch

        if not hasattr(py, "conv2d_backward"):
            self.skipTest("未导出conv2d_backward")
        np.random.seed(3)
        x = np.random.randn(4,3,28,28).astype(np.float32)
        filt = np.random.randn(6,3,3,3).astype(np.float32)
        grad_out = np.random.randn(4,6,28,28).astype(np.float32)
        px = make_tensor(py, x, "gpu")
        pf = make_tensor(py, filt, "gpu")
        gpy = make_tensor(py, grad_out, "gpu")
        grad_in = make_tensor(py, np.zeros_like(x), "gpu")
        grad_filt = make_tensor(py, np.zeros_like(filt), "gpu")
        py.conv2d_backward(gpy, px, pf, grad_in, grad_filt)
        grad_in_np = to_numpy_from_py(grad_in)
        grad_filt_np = to_numpy_from_py(grad_filt)
        # expected grads via torch
        expect_grad_in = F.conv_transpose2d(torch.from_numpy(grad_out), torch.from_numpy(filt), stride=1, padding=1).numpy()

        # 正确计算 weight 的梯度：使用 torch.nn.grad.conv2d_weight
        expect_grad_filt = torch.nn.grad.conv2d_weight(
            input=torch.from_numpy(x),
            weight_size=torch.from_numpy(filt).shape,
            grad_output=torch.from_numpy(grad_out),
            stride=1,
            padding=1
        ).numpy()
        self.assertTrue(np.allclose(grad_in_np, expect_grad_in, atol=1e-3))
        self.assertTrue(np.allclose(grad_filt_np, expect_grad_filt, atol=1e-3))


    def test_maxpool2d_forward(self):
        py = self.py
        F = self.F
        torch = self.torch

        if not hasattr(py, "max_pool2d_forward") and not hasattr(py, "max_pool2d_forward_mask"):
            self.skipTest("未导出 max_pool2d_forward 类函数")
        np.random.seed(4)
        x = np.random.randn(4,3,28,28).astype(np.float32)
        px = make_tensor(py, x, "gpu")
        if hasattr(py, "max_pool2d_forward"):
            out = py.max_pool2d_forward(px)
        else:
            out = py.max_pool2d_forward_mask(px)
        out_np = to_numpy_from_py(out)

        expect = F.max_pool2d(torch.from_numpy(x), kernel_size=2, stride=2).numpy()
        self.assertEqual(out_np.shape, expect.shape)
        self.assertTrue(np.allclose(out_np, expect, atol=1e-3))

    def test_softmax_and_cross_entropy(self):
        py = self.py
        F = self.F
        torch = self.torch

        if not hasattr(py, "softmax_forward"):
            self.skipTest("未导出 softmax_forward")
        if not hasattr(py, "cross_entropy_forward"):
            self.skipTest("未导出 cross_entropy_forward")
        np.random.seed(5)
        logits = np.random.randn(64,10).astype(np.float32)
        labels = np.random.randint(0, 10, size=(64,), dtype=np.int64)

        pl = make_tensor(py, logits, "gpu")
        pout = py.softmax_forward(pl)
        pout_np = to_numpy_from_py(pout)

        expect_softmax = F.softmax(torch.from_numpy(logits), dim=1).numpy()
        self.assertTrue(np.allclose(pout_np, expect_softmax, atol=1e-5))

        lab_t = make_tensor(py, labels.astype(np.int32), "gpu")
        loss = py.cross_entropy_forward(pl, lab_t)
        loss_val = float(loss)

        loss_torch = F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(labels), reduction='mean').item()
        self.assertAlmostEqual(loss_val, loss_torch, places=5)
        

if __name__ == "__main__":
    unittest.main()