import sys
import os
import numpy as np

# Add framework directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../framework"))

import optimizer as nn
from example_utils import (
    load_cifar10,
    CIFAR10Dataset,
    Compose,
    random_crop,
    random_horizontal_flip,
    train_model
)

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
    
    model_name = "SimpleCNN"
    script_name = os.path.basename(__file__)
    augmentation = "RandomCrop+Flip"
    
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
    device = "gpu"
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, device=device),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, device=device),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 10, device=device)
    )
    
    train_model(model, train_dataset, test_dataset, config, model_name, script_name, augmentation, device)

if __name__ == "__main__":
    train()
