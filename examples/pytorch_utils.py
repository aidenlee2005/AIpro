import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os
import numpy as np
import pickle

# Define Models

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class VGG_Custom(nn.Module):
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 32 -> 16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 16 -> 8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ResidualBlock_Custom(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels)
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
        out += identity
        out = self.relu(out)
        return out

class ResNet_Custom(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = nn.Sequential(
            ResidualBlock_Custom(16, 16),
            ResidualBlock_Custom(16, 16)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock_Custom(16, 32),
            ResidualBlock_Custom(32, 32)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock_Custom(32, 64),
            ResidualBlock_Custom(64, 64)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Data Loading
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='bytes')
        X = datadict[b'data']
        Y = datadict[b'labels']
        X = X.reshape(10000, 3, 32, 32).astype("float32")
        Y = np.array(Y, dtype="long")
        return X, Y

class CIFAR10Dataset(Dataset):
    def __init__(self, root_dir, train=True):
        self.root_dir = root_dir
        self.train = train
        self.data = []
        self.targets = []
        
        if self.train:
            for b in range(1, 6):
                f = os.path.join(root_dir, 'data_batch_%d' % (b, ))
                if not os.path.exists(f):
                     f = os.path.join(root_dir, 'cifar-10-batches-py', 'data_batch_%d' % (b, ))
                X, Y = load_cifar_batch(f)
                self.data.append(X)
                self.targets.append(Y)
            self.data = np.concatenate(self.data)
            self.targets = np.concatenate(self.targets)
        else:
            f = os.path.join(root_dir, 'test_batch')
            if not os.path.exists(f):
                f = os.path.join(root_dir, 'cifar-10-batches-py', 'test_batch')
            self.data, self.targets = load_cifar_batch(f)
            
        # Normalize
        self.data = self.data / 255.0
        mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)
        self.data = (self.data - mean) / std
        self.data = self.data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), self.targets[idx]

def run_pytorch_benchmark(model_name, batch_size=64, device='cuda'):
    if not torch.cuda.is_available():
        device = 'cpu'
    
    print(f"Running PyTorch Benchmark for {model_name} on {device}...")
    
    # Setup Data
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    train_dataset = CIFAR10Dataset(data_dir, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Setup Model
    if model_name == 'SimpleCNN':
        model = SimpleCNN()
    elif model_name == 'VGG':
        model = VGG_Custom()
    elif model_name == 'ResNet':
        model = ResNet_Custom()
    else:
        raise ValueError(f"Unknown model: {model_name}")
        
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Warmup
    model.train()
    print("Warmup...")
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if i > 10: break
        
    # Benchmark
    print("Benchmarking...")
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    # Run 1 epoch (or partial if too slow, but 1 epoch is good for comparison)
    # CIFAR10 is small enough.
    for inputs, targets in enumerate(train_loader):
        pass # Just iterate? No, need to run forward/backward
        
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    epoch_time = end_time - start_time
    print(f"PyTorch Epoch Time: {epoch_time:.4f}s")
    return epoch_time

if __name__ == "__main__":
    # Test
    run_pytorch_benchmark('SimpleCNN')
