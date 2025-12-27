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

class FusedVGG(nn.Module):
    def __init__(self, device="gpu"):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.ConvReLU(3, 32, kernel_size=3, stride=1, padding=1, device=device),
            nn.ConvReLU(32, 32, kernel_size=3, stride=1, padding=1, device=device),
            nn.MaxPool2d(kernel_size=2), # 32 -> 16

            # Block 2
            nn.ConvReLU(32, 64, kernel_size=3, stride=1, padding=1, device=device),
            nn.ConvReLU(64, 64, kernel_size=3, stride=1, padding=1, device=device),
            nn.MaxPool2d(kernel_size=2), # 16 -> 8

            # Block 3
            nn.ConvReLU(64, 128, kernel_size=3, stride=1, padding=1, device=device),
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
        'epochs': 50,
        'optimizer': 'SGD'
    }
    
    model_name = "FusedVGG"
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
    model = FusedVGG(device=device)

    train_model(model, train_dataset, test_dataset, config, model_name, augmentation, device)

if __name__ == "__main__":
    train()
