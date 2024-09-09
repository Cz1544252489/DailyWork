import os
import torchvision
from torch.utils.tensorboard import SummaryWriter

# 设置代理
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'https://127.0.0.1:7890'

dataset_transfrom = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, transform=dataset_transfrom, download=True)
test_set = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=dataset_transfrom, download=True)

writer = SummaryWriter("P1")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()


