import os
import time
import argparse
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


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
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=transform)

    train_sampler = DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 测试集只在 rank0 上跑，无需分布式 sampler
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return trainloader, testloader, train_sampler


def train_one_epoch(model, optimizer, criterion, dataloader, device, sampler, epoch):
    model.train()
    sampler.set_epoch(epoch)
    torch.cuda.synchronize()
    start = time.perf_counter()

    running_loss = 0.0
    total_samples = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    torch.cuda.synchronize()
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
        for inputs, labels in dataloader:
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
    parser = argparse.ArgumentParser(description="DDP training for CIFAR-10 LeNet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save", type=str, default="./pytorch/cifar_net_ddp.pth")
    parser.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo"], help="Distributed backend")
    args = parser.parse_args()

    # 常见 vGPU/容器环境下的 NCCL 问题规避
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")  # 禁用 P2P，避免不支持的硬件上崩溃
    os.environ.setdefault("NCCL_IB_DISABLE", "1")   # 如无 IB，禁用以防初始化异常
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")  # 默认禁用 SHM，防止某些容器环境下崩溃
    # 如需调试可临时开启：NCCL_DEBUG=INFO 或设置上述变量为 0

    dist.init_process_group(backend=args.backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    set_seed(42)

    trainloader, testloader, train_sampler = get_dataloaders(args.batch_size, args.num_workers)

    model = LeNet().to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if local_rank == 0:
        print(f"DDP start | world_size={world_size}, batch={args.batch_size}, epochs={args.epochs}")

    for epoch in range(args.epochs):
        loss, elapsed, ips = train_one_epoch(model, optimizer, criterion, trainloader, device, train_sampler, epoch)
        if local_rank == 0:
            print(f"[Epoch {epoch+1}/{args.epochs}] loss={loss:.4f} time={elapsed:.2f}s throughput={ips*world_size:.1f} imgs/s (global)")

    # 只在 rank0 评估和保存
    if local_rank == 0:
        acc, class_acc = evaluate(model.module, testloader, device)
        print(f"Test accuracy: {acc:.2f}%")
        for i, a in enumerate(class_acc):
            print(f"Class {i}: {a:.2f}%")
        torch.save(model.module.state_dict(), args.save)
        print(f"Saved weights to {args.save}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
