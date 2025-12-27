import sys
import os
import numpy as np
import time

# Add framework directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../framework"))

import optimizer as nn
from example_utils import (
    load_cifar10,
    CIFAR10Dataset,
    Compose,
    random_crop,
    random_horizontal_flip,
    get_batch,
    save_model,
    ExperimentLogger,
    progress_wrapper
)

class VGG(nn.Module):
    def __init__(self, device="gpu"):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device),
            nn.BatchNorm2d(64, device=device),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device),
            nn.BatchNorm2d(64, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32 -> 16

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False, device=device),
            nn.BatchNorm2d(128, device=device),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False, device=device),
            nn.BatchNorm2d(128, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16 -> 8

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False, device=device),
            nn.BatchNorm2d(256, device=device),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False, device=device),
            nn.BatchNorm2d(256, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512, device=device),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10, device=device)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train():
    # Configuration
    config = {
        'batch_size': 64,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'epochs': 100,
        'optimizer': 'Adam'
    }
    
    model_name = "VGG_Wide"
    script_name = os.path.basename(__file__)
    augmentation = "RandomCrop+Flip"
    device = "gpu"

    print(f"Running {script_name} with {model_name}...")

    # Load Data
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    X_train, Y_train, X_test, Y_test = load_cifar10(data_dir)

    transforms = Compose([
        lambda x: random_crop(x, padding=4),
        lambda x: random_horizontal_flip(x, p=0.5)
    ])

    train_dataset = CIFAR10Dataset(X_train, Y_train, transform=transforms, mode='train')
    test_dataset = CIFAR10Dataset(X_test, Y_test, transform=None, mode='test')

    # Define Model
    model = VGG(device=device)
    
    # Optimizer setup
    optimizer = nn.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    total_start_time = time.time()
    total_batches = (len(train_dataset) + config['batch_size'] - 1) // config['batch_size']

    for epoch in range(config['epochs']):
        # Learning Rate Decay
        if epoch == 50 or epoch == 80:
            optimizer.lr *= 0.1
            print(f"Decaying learning rate to {optimizer.lr}")

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
    from pytorch_utils import run_pytorch_benchmark
    try:
        pytorch_time = run_pytorch_benchmark(model_name, batch_size=config['batch_size'])
    except Exception as e:
        print(f"PyTorch benchmark failed: {e}")
        pytorch_time = None
        
    # Log
    logger = ExperimentLogger()
    logger.log(model_name, augmentation, config, tr_acc, te_acc, total_time, pytorch_time)
    
    # Save Model
    save_model(model, model_name, te_acc)

if __name__ == "__main__":
    train()
