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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device="gpu"):
        super().__init__()
        # bias=False because BN follows
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(out_channels, device=device)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn2 = nn.BatchNorm2d(out_channels, device=device)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, device=device),
                nn.BatchNorm2d(out_channels, device=device)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = out + identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10, device="gpu"):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, device=device)
        self.bn1 = nn.BatchNorm2d(16, device=device)
        self.relu = nn.ReLU()
        
        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16, device=device),
            ResidualBlock(16, 16, device=device)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32, device=device),
            ResidualBlock(32, 32, device=device)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64, device=device),
            ResidualBlock(64, 64, device=device)
        )
        
        self.avgpool = nn.GlobalAvgPool()
        self.fc = nn.Linear(64, num_classes, device=device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global Average Pooling
        x = self.avgpool(x)
        
        x = self.fc(x)
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
    
    model_name = "ResNet"
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
    model = ResNet(device=device)
    
    train_model(model, train_dataset, test_dataset, config, model_name, script_name, augmentation, device)

if __name__ == "__main__":
    train()
