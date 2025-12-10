import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm


batch_size = 4

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
criterion = nn.CrossEntropyLoss()

writer = {
    'momentum=0.9' : SummaryWriter('runs/momentum/0.9'),
    'momentum=0.5' : SummaryWriter('runs/momentum/0.5'),
    'momentum=0.0' : SummaryWriter('runs/momentum/0.0')
}

if __name__ == '__main__':
    for i in [0.5, 0.0]:
        net = LeNet()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=i)
        for epoch in range(10):
            running_loss = 0.0
            for data in tqdm(trainloader,desc=f'Epoch {epoch+1}'):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'[Epoch {epoch + 1}] loss: {running_loss / len(trainloader):.3f}')
            writer['momentum='+str(i)].add_scalar('training loss', running_loss / len(trainloader), epoch+1)
            
        print('Finished Training.')        
