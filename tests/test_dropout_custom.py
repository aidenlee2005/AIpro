import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import framework.tensor as tensor
import framework.operators as ops
import framework.optimizer as optim
import numpy as np

def test_dropout():
    print("Testing Dropout...")
    
    # Test 1: Forward pass randomness
    print("Test 1: Forward pass randomness")
    x_np = np.ones((10, 10), dtype=np.float32)
    x = tensor.TensorFull(x_np, device="gpu")
    
    dropout = optim.Dropout(p=0.5)
    dropout.train() # Set to training mode
    
    y1 = dropout(x)
    y2 = dropout(x)
    
    y1_np = y1.numpy()
    y2_np = y2.numpy()
    
    # Check if outputs are different (randomness)
    if np.allclose(y1_np, y2_np):
        print("Warning: Two consecutive dropout calls produced identical results. This might happen by chance but is unlikely for large tensors.")
    else:
        print("Pass: Randomness observed.")
        
    # Check if values are scaled correctly (Inverted Dropout)
    # Expected values are either 0 or 1/(1-p) = 2
    unique_vals = np.unique(y1_np)
    print(f"Unique values in output: {unique_vals}")
    if np.all(np.isin(unique_vals, [0, 2])):
        print("Pass: Output values are correct (0 or 2).")
    else:
        print("Fail: Output values are incorrect.")

    # Test 2: Eval mode
    print("\nTest 2: Eval mode")
    dropout.eval()
    y_eval = dropout(x)
    if np.allclose(y_eval.numpy(), x_np):
        print("Pass: Eval mode returns input unchanged.")
    else:
        print("Fail: Eval mode modified input.")

    # Test 3: Backward pass
    print("\nTest 3: Backward pass")
    dropout.train()
    x = tensor.TensorFull(x_np, requires_grad=True, device="gpu")
    y = dropout(x)
    y.sum().backward()
    
    grad = x.grad.numpy()
    # Gradient should be mask * scale
    # Since input was all ones, output is mask * scale * 1
    # Gradient of sum(output) w.r.t output is all ones
    # So gradient w.r.t input should be mask * scale
    
    if np.allclose(grad, y.numpy()):
        print("Pass: Gradient matches output pattern (for all-ones input).")
    else:
        print("Fail: Gradient check failed.")

if __name__ == "__main__":
    test_dropout()
