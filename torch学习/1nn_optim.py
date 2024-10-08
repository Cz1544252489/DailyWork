import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../CIFAR10", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=32)

class Zhuo(nn.Module):
    def __init__(self):
        super(Zhuo, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64,10)
        )

    def forward(self,x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
zhuo = Zhuo()
optim = torch.optim.SGD(zhuo.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = zhuo(imgs)
        result_loss = loss(outputs, targets)

        optim.zero_grad()
        result_loss.backward()
        optim.step()

        running_loss = running_loss + result_loss

    print(running_loss)

