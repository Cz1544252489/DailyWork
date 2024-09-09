import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


test_data = torchvision.datasets.CIFAR10("../CIFAR10", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset= test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

# 测试数据集
# img, target = test_data[0]
# print(img.shape)
# print(target)

writer = SummaryWriter("P2")
for epoch in range(10):
    step = 0
    for data in test_loader:
        imgs, targets =data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()