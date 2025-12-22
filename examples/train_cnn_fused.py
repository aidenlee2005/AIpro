import sys
import os
import time
import numpy as np

# Add framework directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../framework"))

import optimizer as nn
from example_utils import load_cifar10, CIFAR10Dataset, get_batch, ExperimentLogger, save_model, progress_wrapper
from pytorch_utils import run_pytorch_benchmark

def train():
    # Configuration
    config = {
        'batch_size': 64,
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'epochs': 5,
        'optimizer': 'SGD'
    }
    
    model_name = "SimpleCNN_Fused"
    script_name = os.path.basename(__file__)
    augmentation = "None"
    
    print(f"Running {script_name} with {model_name}...")
    
    # Load Data
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    X_train, Y_train, X_test, Y_test = load_cifar10(data_dir)
    
    train_dataset = CIFAR10Dataset(X_train, Y_train, transform=None, mode='train')
    test_dataset = CIFAR10Dataset(X_test, Y_test, transform=None, mode='test')
    
    # Define Model
    device = "gpu"
    model = nn.Sequential(
        nn.ConvReLU(3, 16, kernel_size=3, stride=1, padding=1, device=device),
        nn.MaxPool2d(kernel_size=2),
        nn.ConvReLU(16, 32, kernel_size=3, stride=1, padding=1, device=device),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10, device=device)
    )
    
    optimizer = nn.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    total_start_time = time.time()
    
    total_batches = (len(train_dataset) + config['batch_size'] - 1) // config['batch_size']

    for epoch in range(config['epochs']):
        model.train()
        epoch_start = time.time()
        correct = 0
        total = 0
        loss_sum = 0
        step_start = time.time()
        
        for batch_idx, (batch_x, batch_y) in progress_wrapper(
            get_batch(train_dataset, config['batch_size'], device=device),
            total_batches,
            prefix=f"Epoch {epoch+1}/{config['epochs']}",
            start_time=step_start,
        ):
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            # Metrics
            pred = output.numpy().argmax(axis=1)
            correct += (pred == batch_y.numpy()).sum()
            total += batch_y.shape[0]
            loss_sum += loss.numpy()

            # tqdm handles display; fallback handled in progress_wrapper
            
        epoch_time = time.time() - epoch_start
        tr_acc = correct / total
        print(f"Epoch {epoch+1}/{config['epochs']} - Time: {epoch_time:.2f}s - Loss: {loss_sum/total:.4f} - Acc: {tr_acc:.4f}")
        
    total_time = time.time() - total_start_time
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    for batch_x, batch_y in get_batch(test_dataset, config['batch_size'], device=device):
        output = model(batch_x)
        pred = output.numpy().argmax(axis=1)
        correct += (pred == batch_y.numpy()).sum()
        total += batch_y.shape[0]
    
    te_acc = correct / total
    print(f"Test Accuracy: {te_acc:.4f}")
    
    # PyTorch Benchmark
    try:
        pytorch_time = run_pytorch_benchmark(model_name, batch_size=config['batch_size'])
    except Exception as e:
        print(f"PyTorch benchmark failed: {e}")
        pytorch_time = None
        
    # Log
    logger = ExperimentLogger()
    logger.log(script_name, model_name, augmentation, config, tr_acc, te_acc, total_time, pytorch_time)
    
    # Save Model
    save_model(model, model_name, te_acc)

if __name__ == "__main__":
    train()
