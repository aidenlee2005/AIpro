import torch
import train

testset = train.testset
testloader = train.testloader
net = train.LeNet()
net.load_state_dict(torch.load('./cifar_net.pth',weights_only=True))

if __name__ == '__main__':
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')