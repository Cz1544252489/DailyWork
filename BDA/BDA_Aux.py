import argparse
import os
import sys
import time

import torch
import torchvision
from torch import nn
from torch.utils.data import Subset, DataLoader, Dataset

from geoopt.manifolds import Stiefel


# parser
def parser():
    parser = argparse.ArgumentParser(description="Nothing")
    parser.add_argument('--proxy', type=bool, default=False, help="是否使用代理下载文件")
    parser.add_argument('--logdir', type=str, default="test",help="文件保存的位置，为空的话，不保存内容。")
    parser.add_argument('-N','--batch_size', type=int, default=30, help="batch_size")
    parser.add_argument('--niu', type=float, default= 0.5)
    parser.add_argument('--niu1', type=float, default=0.001)
    parser.add_argument('--niu2', type=float, default=0.0001)
    parser.add_argument('--s_l', type=float, default=0.1)
    parser.add_argument('--s_u', type=float, default=0.01)
    parser.add_argument('--lr_x', type=float, default=0.01)
    parser.add_argument('--lr_y', type=float, default=0.01)
    parser.add_argument('-E','--epochs', type=int, default=10)
    parser.add_argument('-K','--K',type= int, default= 5)
    parser.add_argument('--t1',type=float, default=0.25)
    parser.add_argument('--t2', type=float, default=1.0)
    parser.add_argument('--flag', type=str, default="Euclidean",
                        help="有三种选择，'Euclidean','Stiefel和'Hybrid'.")
    parser.add_argument('--rand_seed',type=int, default=123456)
    parser.add_argument('--seed_option', type=str, default="time",
                        help="可用设置为'time'，或者为'fixed'固定值；后者可用设置'rand_seed'")
    parser.add_argument('--datasets', type=str, default="MNIST",
                        help="目前可用'MNIST'(默认值)和'FMNIST'(即FashionMNIST)")
    parser.add_argument('--pollute', type=bool, default=True,
                        help="设置是否污染训练数据，为布尔值，默认为是")
    parser.add_argument('--output', type=str, default="epochs+norm_of_grad",
                        help="设置循环时输出的内容，'epochs'输出循环进度,'norm_of_grad'输出梯度的二范数")
    args = parser.parse_args()
    return args

# 小函数

def set_random_set(args):
    if args.seed_option == "time":
        rand_seed = int(time.time())
    elif args.seed_option == "fixed":
        rand_seed = args.rand_seed
    else:
        sys.exit("Wrong Input for rand_seed!")

    torch.manual_seed(rand_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rand_seed)

# 一个类
class Paras:
    def __init__(self, device, niu, niu1, niu2, s_u, s_l, lr_x, lr_y, N, t1, t2):
        self.device = device
        self.niu = torch.tensor(niu).to(device)
        self.niu1 = torch.tensor(niu1).to(device)
        self.niu2 = torch.tensor(niu2).to(device)
        self.s_u = torch.tensor(s_u).to(device)
        self.s_l = torch.tensor(s_l).to(device)
        self.lr_x = torch.tensor(lr_x).to(device)
        self.lr_y = torch.tensor(lr_y).to(device)
        self.N = torch.tensor(N).to(device)
        self.t1 = torch.tensor(t1).to(device)
        self.t2 = torch.tensor(t2).to(device)

    def alpha(self, k):
        output = self.t1/k
        return output

    def beta(self, k):
        output = self.t2/k
        return output

# 数据导入
class PollutedMNIST(Dataset):
    def __init__(self, mnist_dataset, indices_to_pollute, pollution_rate=0.5):
        self.mnist_dataset = mnist_dataset
        self.indices_to_pollute = set(indices_to_pollute)
        self.pollution_rate = pollution_rate
        self.random_errors = torch.randint(10, 20, (len(indices_to_pollute),))  # 错误标签范围10到19

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        data, label = self.mnist_dataset[idx]
        if idx in self.indices_to_pollute:
            error_idx = idx % len(self.random_errors)  # 重复使用错误标签以匹配索引
            label = self.random_errors[error_idx]
        return data, label

def data_load(batch_size, datasets_type, pollute, proxy):
    if proxy is True:
        os.environ['http_proxy'] = 'http://127.0.0.1:7890'
        os.environ['https_proxy'] = 'https://127.0.0.1:7890'

    if datasets_type == "MNIST":
        data1 = torchvision.datasets.MNIST("../MNIST", train=True, transform=torchvision.transforms.ToTensor(),
                                           download=True)
        data2 = torchvision.datasets.MNIST("../MNIST", train=False, transform=torchvision.transforms.ToTensor(),
                                           download=True)
    elif datasets_type == "FMNIST":
        data1 = torchvision.datasets.FashionMNIST("../FMNIST", train=True, transform=torchvision.transforms.ToTensor(),
                                                  download=True)
        data2 = torchvision.datasets.FashionMNIST("../FMNIST", train=False, transform=torchvision.transforms.ToTensor(),
                                                  download=True)
    else:
        raise ValueError("输入错误！datasets_type必须是'MNIST','FMNIST'(即Fashion MNIST)")

    train_data_size = 5000
    validation_data_size = 5000
    test_data_size = 60000

    indices = torch.randperm(train_data_size + validation_data_size)
    indices1 = indices[:train_data_size]
    indices2 = indices[train_data_size:]

    train_data = Subset(data2, indices1)
    validation_data = Subset(data2, indices2)
    test_data = data1

    if pollute is True:
        indices_to_pollute = torch.randperm(train_data_size)[:2500]
        polluted_train_data = PollutedMNIST(train_data, indices_to_pollute)
        train_loader = DataLoader(polluted_train_data, batch_size=batch_size)
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size)

    validation_loader = DataLoader(validation_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, validation_loader, test_loader


# 模型部分

class ModelY(nn.Module):
    def __init__(self, device, flag="Euclidean"):
        super().__init__()
        self.device = device
        self.flag = flag
        y_size = [785, 10]
        M = Stiefel()
        self.M = M
        y = torch.randn(y_size)
        if self.flag in ["Stiefel", "Hybrid"]:
            y = M.random_naive(y_size)
            y = y.detach()
        elif self.flag not in ["Euclidean"]:
            raise ValueError("输入错误！flag必须是'Euclidean', 'Stiefel'或'Hybrid'之一")
        y = y.to(device)
        self.y = nn.Parameter(y)

    def forward(self, u):
        u_size = u.shape[0]
        prod = torch.cat((u.view(u_size, -1), torch.ones([u_size, 1], device=self.device)), 1)
        return torch.mm(prod, self.y)

    def update_man(self, lr, direction):
        with torch.no_grad():
            new_y = self.y - lr * direction
            if self.flag in ["Stiefel"]:
                new_y = self.M.retr(self.y, - lr * direction)
            elif self.flag not in ["Euclidean", "Hybrid"]:
                raise ValueError("输入错误！flag必须是'Euclidean', 'Stiefel'或'Hybrid'之一")
            self.y = nn.Parameter(new_y)

    def update_opt(self, optimizer, direction):
        self.y.grad = direction
        optimizer.step()
        optimizer.zero_grad()

class ModelX(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        x = torch.abs(torch.randn([1, 5000], requires_grad=True))
        self.x = nn.Parameter(x.to(device))

    def update_man(self, lr, direction):
        with torch.no_grad():
            new_x = self.x - lr * direction
            self.x = nn.Parameter(new_x)

    def update_opt(self, optimizer, direction):
        self.x.grad = direction
        optimizer.step()
        optimizer.zero_grad()

# 损失函数
class Loss_F:
    def __init__(self, device, flag):
        self.device = device
        CroEnt = nn.CrossEntropyLoss()
        self.CroEnt = CroEnt.to(device)
        self.flag = flag

    def exec(self, dataloader, model_x, model_y, P):
        step = 0
        total_loss = 0
        for data in dataloader:
            u, v = data
            u = u.to(self.device)
            v = v.to(self.device)
            outputs = model_y(u)

            l2_x = torch.pow(torch.norm(model_x.x, p=2), 2)
            if self.flag in ["Stiefel"]:
                l2_y = 0
            elif self.flag in ["Euclidean", "Hybrid"]:
                l2_y = torch.pow(torch.norm(model_y.y, p=2), 2)
            else:
                raise ValueError("类型输入错误")
            loss = self.CroEnt(outputs, v) + P.niu1 * l2_x + l2_y
            step = step + 1
            total_loss = total_loss + loss
        return total_loss


class Loss_f:
    def __init__(self, device):
        self.device = device
        CroEnt = nn.CrossEntropyLoss(reduction='none')
        self.CroEnt = CroEnt.to(device)

    def exec(self, dataloader, model_x, model_y, P):
        step = 0
        total_loss = 0
        for data in dataloader:
            u, v = data
            u = u.to(self.device)
            v = v.to(self.device)
            outputs = model_y(u)

            loss = torch.sum(
                torch.mul(torch.sigmoid(model_x.x[:, P.N * step:P.N * (step + 1)]), self.CroEnt(outputs, v)))
            step = step + 1
            total_loss = total_loss + loss
        return total_loss

#
def get_direction(gF_y, gf_y, par, k):
    direction = par.niu * par.alpha(k + 1) * par.s_u * gF_y + (1 - par.niu) * par.beta(k + 1) * par.s_l * gf_y
    return direction


# 测试正确率

def acc(model,loader,device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            u, v = data
            u = u.to(device)
            v = v.to(device)
            outputs = model(u)
            _, predicted = torch.max(outputs, 1)
            total += v.size(0)
            correct += (predicted == v).sum().item()

    accuracy = 100 * correct / total
    return accuracy