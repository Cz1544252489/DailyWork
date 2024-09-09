import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10("../CIFAR10", train=False, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class Zhuo(nn.Module):
    def __init__(self):
        super(Zhuo, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
        self.Maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)
        self.relu = ReLU()

    def forward(self,x):
        x = self.Maxpool1(x)
        return self.relu(x)

zhuo = Zhuo()

writer = SummaryWriter("./logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = zhuo(imgs)
    # print(imgs.shape)
    # print(output.shape)

    writer.add_images("input", imgs, step)

    # output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output",output, step)

    step = step + 1

writer.close()