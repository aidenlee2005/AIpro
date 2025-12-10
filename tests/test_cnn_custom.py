import numpy as np
import sys
import os

# Add framework directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../framework"))

from tensor import TensorFull as Tensor
import optimizer as nn
import operators as F

def test_cnn_forward_backward():
    print("Testing CNN Forward and Backward...")
    
    # Setup device
    device = "gpu"

    # Create a simple model
    # Input: (N, 1, 28, 28) -> Conv2d -> ReLU -> MaxPool -> Flatten -> Linear -> Output
    # Note: Conv2d underlying implementation assumes stride=1, padding=0 (valid convolution)
    # MaxPool underlying implementation assumes kernel=2, stride=2
    
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, device=device), # Output: (N, 4, 28, 28)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), # Output: (N, 4, 14, 14)
        nn.Flatten(), # Output: (N, 4*14*14) = (N, 784)
        nn.Linear(784, 10, device=device)
    )
    
    # Create dummy input
    batch_size = 2
    x_np = np.random.randn(batch_size, 1, 28, 28).astype(np.float32)
    x = Tensor(x_np, device=device)
    
    # Create dummy target (one-hot)
    y_np = np.zeros((batch_size, 10), dtype=np.float32)
    y_np[0, 3] = 1.0
    y_np[1, 7] = 1.0
    y = Tensor(y_np, device=device)
    
    # Optimizer
    optimizer = nn.SGD(model.parameters(), lr=1e-5, momentum=0.0)
    
    # Training loop (1 step)
    model.train()
    optimizer.zero_grad()
    
    # Forward
    print("Running forward pass...")
    x1 = model.modules_list[0](x) # Conv2d
    print("Conv2d out shape:", x1.shape)
    x2 = model.modules_list[1](x1) # ReLU
    x3 = model.modules_list[2](x2) # MaxPool
    print("MaxPool out shape:", x3.shape)
    x4 = model.modules_list[3](x3) # Flatten
    print("Flatten out shape:", x4.shape)
    logits = model.modules_list[4](x4) # Linear
    print("Logits shape:", logits.shape)
    
    # logits = model(x) # Already computed manually above

    print("Logits sample:", logits.numpy()[0])
    print("Y sample:", y.numpy()[0])
    
    # Loss (MSE for simplicity)
    # MSE = mean((logits - y)^2)
    diff = logits - y
    loss = (diff * diff).sum() / Tensor(np.array(batch_size * 10, dtype=np.float32), device=device)
    
    print("Loss:", loss.numpy())
    
    # Backward
    print("Running backward pass...")
    loss.backward()
    
    # Check gradients
    print("Checking gradients...")
    has_grad = False
    params_with_grad = 0
    total_params = len(model.parameters())
    
    for i, p in enumerate(model.parameters()):
        if p.grad is not None:
            has_grad = True
            params_with_grad += 1
            # print(f"Param {i} grad shape: {p.grad.shape}")
        else:
            print(f"Parameter {i} missing grad!")
            
    print(f"Params with grad: {params_with_grad}/{total_params}")
    
    if params_with_grad == total_params:
        print("All parameters have gradients.")
    else:
        print("Some parameters are missing gradients!")
        
    # Step
    optimizer.step()
    print("Optimizer step executed.")
    
    # Forward again to check loss decrease
    logits_new = model(x)
    diff_new = logits_new - y
    loss_new = (diff_new * diff_new).sum() / Tensor(np.array(batch_size * 10, dtype=np.float32))
    print("New Loss:", loss_new.numpy())
    
    if loss_new.numpy() < loss.numpy():
        print("Test Passed: Loss decreased.")
    else:
        print("Test Failed: Loss did not decrease.")

if __name__ == "__main__":
    try:
        test_cnn_forward_backward()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Test Failed with exception.")
