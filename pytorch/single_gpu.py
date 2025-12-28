import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_dataloaders(batch_size: int, num_workers: int):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 复用上级目录已存在的数据集，避免重复下载
    data_root = os.path.join(os.path.dirname(__file__), "..", "data")
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return trainloader, testloader


def train_one_epoch(model, optimizer, criterion, dataloader, device, epoch):
    model.train()
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()

    running_loss = 0.0
    total_samples = 0
    for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - start
    avg_loss = running_loss / total_samples
    imgs_per_sec = total_samples / elapsed if elapsed > 0 else 0.0
    return avg_loss, elapsed, imgs_per_sec


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    class_total = [0] * 10
    class_correct = [0] * 10
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for p, l in zip(predicted, labels):
                idx = int(l.item())
                class_total[idx] += 1
                if p == l:
                    class_correct[idx] += 1

    class_acc = [100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 for i in range(10)]
    return 100.0 * correct / total, class_acc


def main():
    parser = argparse.ArgumentParser(description="Single-GPU training for CIFAR-10 LeNet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save", type=str, default="./pytorch/cifar_net.pth")
    parser.add_argument("--logdir", type=str, default="pytorch/runs/cifar10_single")
    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = get_dataloaders(args.batch_size, args.num_workers)

    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    writer = SummaryWriter(args.logdir)

    print(f"Start training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    for epoch in range(args.epochs):
        loss, elapsed, ips = train_one_epoch(model, optimizer, criterion, trainloader, device, epoch)
        print(f"[Epoch {epoch+1}/{args.epochs}] loss={loss:.4f} time={elapsed:.2f}s throughput={ips:.1f} imgs/s")
        writer.add_scalar("train/loss", loss, epoch + 1)
        writer.add_scalar("train/throughput", ips, epoch + 1)

    acc, class_acc = evaluate(model, testloader, device)
    print(f"Test accuracy: {acc:.2f}%")
    for i, a in enumerate(class_acc):
        print(f"Class {i}: {a:.2f}%")
    writer.add_scalar("test/accuracy", acc, args.epochs)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save(model.state_dict(), args.save)
    print(f"Saved weights to {args.save}")


if __name__ == "__main__":
    main()