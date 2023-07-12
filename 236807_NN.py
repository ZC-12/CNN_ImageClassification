import torch 
import torch.nn as nn 
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import sys

def imgshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=4, stride=2, padding=1) # 32*32*3 --> 16*16*8
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1) # 16*16*8 --> 8*8*16
        # self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.fc = nn.Linear(64, 10)
        # self.dropout = nn.Dropout(0.2)
        
        # hyperparameter tune
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=2, stride=2) # 32*32*3 --> 32*32*64
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1) # 32*32*64 --> 16*16*128
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1) # 16*16*128 --> 8*8*256
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return x

def main():
    # download dataset
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    trainset=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
    testset=torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
    
    # define hyperparameters
    batch_size = 10
    lr = 0.01
    epoch = 25

    # initialize
    trainloader=torch.utils.data.DataLoader(trainset,batch_size,shuffle=True,num_workers=2)
    testloader=torch.utils.data.DataLoader(testset,batch_size,shuffle=False,num_workers=2)
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)
    
    # training
    print("Start Training")
    l = []
    net.train()
    for j in range(epoch): 
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            l.append(loss.item())
            if i % 2000 == 1999:
                print(f'[{j + 1}, {i + 1:5d}] loss: {running_loss / 2000:.4f}')
                running_loss = 0.0
    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the training images: %.4f' % (correct/total))

    # testing
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the testing images: %.4f' % (correct/total))

    plt.plot(range(len(l)),l)
    plt.show()

if __name__=='__main__':
    sys.exit(main())
