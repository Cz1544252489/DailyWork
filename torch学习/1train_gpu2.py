import torch
import torchvision.datasets
from scipy.signal import ellip
from torch import nn
from torch.nn import Sequential, Linear, Flatten, MaxPool2d, Conv2d
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torch.xpu import device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = torchvision.datasets.CIFAR10("../CIFAR10", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("../CIFAR10", train=False, transform=torchvision.transforms.ToTensor())


train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_loader = DataLoader(dataset= train_data, batch_size=32)
test_loader = DataLoader(dataset= test_data, batch_size=32)

# 创建模型

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
zhuo = Zhuo()
zhuo = zhuo.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.01
optim = torch.optim.SGD(zhuo.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0

total_test_step = 0

epoch = 10

writer = SummaryWriter("./train1")

for i in range(epoch):
    print("-------第{}轮训练开始了------".format(i+1))

    zhuo.train()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = zhuo(imgs)
        loss = loss_fn(outputs, targets)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 ==0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    zhuo.eval()
    total_test_loss = 0
    total_accurary = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = zhuo(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accurary = (outputs.argmax(1)==targets).sum()
            total_accurary = total_accurary + accurary

    print("整体数据集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accurary/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accurary", total_accurary/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(zhuo,"zhuo_{}.pth".format(i))
    # torch.save(zhuo.state_dict(), "zhuo_{}.pth".format(i))
    print("模型已保存")

writer.close()


