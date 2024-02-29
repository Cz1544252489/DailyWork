##  用python写算法

### 最速梯度下降

```python
# 一个简单的梯度法的例子
import numpy as np
from numpy import linalg as LA

m = 100
n = 100
rng = np.random.default_rng()
x = rng.random([m,n])  # 初始值
A = rng.random([m,n])  # 第一个参数

def f(x,A):
    return LA.norm(A-x,'fro')**2

def nabla_f(x,A):
    return 2*(A-x)

def stepsize(x,A,eta):
    return -np.trace(np.transpose(eta)*(x-A))/LA.norm(eta)**2
    
print(f(x,A))
for i in np.arange(500):
    eta = -nabla_f(x,A)
    ss = stepsize(x,A,eta)
    x = x + ss*eta
    print(f(x,A))

```

### 反向传播的基本使用

```python
import torch

# 我们建立一个tensor，并设置requires_grad=True让pytorch跟踪它的计算历史
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 对tensor进行操作
y = x + 2
print(y)

# y是计算的结果，所以它有grad_fn属性
print(y.grad_fn)

# 在y上进行更多操作
z = y * y * 3
out = z.mean()
print(z, out)

# 用 .backward() 计算梯度
out.backward()

#打印梯度
print(x.grad)
```

### 反向传播优化问题

```python
import torch
import torch.optim as optim
var1 = torch.rand([5,5],requires_grad=True)

print(var1)
optimizer = optim.SGD([var1], lr=0.001)

for i in range(5000):
    optimizer.zero_grad()
    loss = torch.norm(var1,'fro')
    if i % 50 == 0:
        print(loss)
    loss.backward()
    optimizer.step()

```

### 使用反向传播解决自定义的问题

```python
import torch
import torch.optim as optim

m = 100
n = 100
x = torch.rand([m,n],requires_grad=True)
A = torch.rand([m,n])

# optimizer = optim.SGD([x], lr=0.01)
optimizer = optim.Adam([x], lr=0.01)

def f(x,A):
    return torch.norm(A-x,'fro')**2

for i in range(500):
    optimizer.zero_grad()
    obj_val = f(x,A)
    if i%10==0:
        print(obj_val)
    obj_val.backward()
    optimizer.step()

```

### 更加复杂的代码

```python
import sys
import idx2numpy
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

def one_hot(label):
    # 用于把标签变成one-hot格式
    m = len(label)
    label_one_hot = torch.zeros([m,10])
    for i in range(m):
        label_one_hot[i,label[i]] = 1
    return label_one_hot

def load_data1(flag,N):
    # 用于导入csv数据，但是目前csv数据的mnist数据集在kaggle上被删掉了
    if flag=='train':
        test = pd.read_csv('mnist/train.csv').to_numpy()
        test = torch.from_numpy(test)
        test = test[1:N+1]
        label = one_hot(test[:,0]).float()
        image = test[:,1:].float()

    if flag=='test':
        test = pd.read_csv('mnist/test.csv').to_numpy()
        test = torch.from_numpy(test)
        test = test[1:N+1]
        label = one_hot(test[:,0]).float()
        image = test[:,1:].float()
    return image,label
    
def load_idx(images_path, labels_path):
    images = idx2numpy.convert_from_file(images_path)
    labels = idx2numpy.convert_from_file(labels_path)
    return images, labels
    
def load_data(flag,N):
    if flag=='train':
        if N>60000:
            sys.exit('Over 6e4 !')
        length = 60000
        images_path = 'mnist/train-images-idx3-ubyte'
        labels_path = 'mnist/train-labels-idx1-ubyte'
        
    if flag=='test':
        if N>10000:
            sys.exit('Over 1e4 !')
        length = 10000
        images_path = 'mnist/t10k-images-idx3-ubyte'
        labels_path = 'mnist/t10k-labels-idx1-ubyte'
        
    images, labels = load_idx(images_path, labels_path)
    images = images.reshape(length,-1)
    images_copy = np.copy(images[0:N])
    labels_copy = np.copy(labels[0:N])
    image = torch.from_numpy(images_copy).float()
    label = one_hot(torch.from_numpy(labels_copy)).float()
    return image, label

def test(hat_label,label):
    ss = nn.Softmax(1)
    aab = torch.sum(abs(ss(hat_label)-label),axis=1)
    accuracy = torch.sum(abs(aab)<=1e-5)/N
    return accuracy

```

```python
p = 28
C = 10
N = 5000

image,label = load_data('train',N)
print(type(image),image.shape)
print(type(label),label.shape)
W = torch.rand([p**2,C],requires_grad=True)
b = torch.rand([N,C],requires_grad=True)
la = torch.rand([N,1],requires_grad=True)
CrossEntropy = nn.CrossEntropyLoss()
ss = nn.Softmax(dim=1)

optimizer = optim.Adam([W,b],lr=1e-2)
optimizer1 = optim.Adam([la],lr=1e-2)
for i in range(300):
    optimizer.zero_grad()
    hat_label = torch.mm(image,W)+b
    
    loss = CrossEntropy(hat_label, label)
    if i % 10 == 0:
        # print(test(hat_label,label))
        print(loss)
    loss.backward()
    optimizer.step()
    
```

