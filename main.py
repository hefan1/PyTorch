import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import myDataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(trainloader):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d,%5d] loss: %.3f'%
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss=0.0

    print('Finished Training')
    return net

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

training=False
PATH = './cifar_net.pth'
dataloader=myDataLoader.trainloader
classes = myDataLoader.classes
testloader=myDataLoader.testloader

if __name__=="__main__":
    if training:
        net=train(dataloader)
        torch.save(net.state_dict(), PATH)
        print('Net parameters saved in '+PATH)
    else:
        dataiter=iter(testloader)
        images,labels=dataiter.next()
        imshow(torchvision.utils.make_grid(images))
        print("GroudTruth: ",
              " ".join('%5s' % classes[labels[j]] for j in range(4)))

        net=Net()
        net.load_state_dict(torch.load(PATH))
        outputs=net(images)
        _,predicted=torch.max(outputs,1)

        print("Predicted: ",
              " ".join("%5s"%(classes[predicted[j]])
                       for j in range(4)))

        correct=0
        total=0
        with torch.no_grad():
            for data in testloader:
                images,labels=data
                outputs=net(images)
                _,predicted=torch.max(outputs,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

