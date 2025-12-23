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

class VGG(nn.Module):
    def __init__(self, device="gpu"):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            # bias=False because BN follows
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False, device=device),
            nn.BatchNorm2d(32, device=device),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False, device=device),
            nn.BatchNorm2d(32, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32 -> 16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device),
            nn.BatchNorm2d(64, device=device),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False, device=device),
            nn.BatchNorm2d(64, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 16 -> 8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False, device=device),
            nn.BatchNorm2d(128, device=device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256, device=device),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10, device=device)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train():
    # Configuration
    config = {
        'batch_size': 64,
        'lr': 0.01,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'epochs': 5,
        'optimizer': 'SGD'
    }
    
    model_name = "VGG"
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
    model = VGG(device=device)

    train_model(model, train_dataset, test_dataset, config, model_name, script_name, augmentation, device)

if __name__ == "__main__":
    train()
