import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Linear, Flatten, MaxPool2d, Conv2d
from torch.utils.data import DataLoader

from model_zhuo import *
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10("../CIFAR10", train=True, transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("../CIFAR10", train=False, transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

batch_size = 1
train_loader = DataLoader(dataset= train_data, batch_size=batch_size)
test_loader = DataLoader(dataset= test_data, batch_size=batch_size)

# 创建模型
zhuo = Zhuo()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

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
        outputs = zhuo(imgs)
        print([outputs.shape, targets.shape])
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


