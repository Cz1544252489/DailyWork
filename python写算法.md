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

