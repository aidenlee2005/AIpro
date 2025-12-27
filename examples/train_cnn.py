import sys
import os
import numpy as np

# Add framework directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../framework"))

import optimizer as nn
from example_utils import (
    load_cifar10,
    CIFAR10Dataset,
    train_model
)

def train():
    # Configuration
    config = {
        'batch_size': 64,
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0,
        'epochs': 50,
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
    
    train_model(model, train_dataset, test_dataset, config, model_name, augmentation, device)

if __name__ == "__main__":
    train()
