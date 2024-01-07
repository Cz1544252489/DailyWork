## Python中重要包的使用

### 安装脚本

```bash
pip install torch
pip install numpy
pip install tensorflow
pip install kaggle  # 用于下载部分数据
pip install pandas  # 用于导入数据
pip install scipy
pip install scikit-learn
pip install matplotlib # 用于画图
pip install torchvision
# 用于构建一个可以远程在网页上运行代码的环境，应该已经安装完成了
# pip install jupyter 
```

可以在 [https://file.cz123.top/9others/CodesFile/requirements.txt](https://file.cz123.top/9others/CodesFile/requirements.txt)找到

```
torch
numpy
tensorflow
kaggle
pandas
scipy
scikit-learn
matplotlib
torchvision
```

那么可以用以下代码执行

```bash
# 进一步整合代码，以后只用打开此软件即可
wget https://file.cz123.top/9others/CodesFile/ToJupyter.sh && bash ToJupyter.sh

wget https://file.cz123.top/9others/CodesFile/requirements.txt
pip install -r requirements.txt
```

### 常用链接

[数据类型转换](https://www.runoob.com/python3/python3-type-conversion.html) [数学函数](https://www.runoob.com/python3/python3-number.html)

### [numpy](https://www.runoob.com/numpy/numpy-ndarray-object.html)

```python
import numpy as np
a=np.array([[1,2,3],[4,5,6]])
print(a)
b=a.reshape((6,))
print(b)
b[0]=100
print(a)

x = np.arange(5, dtype=float)
y = np.arange(5, dtype=float).reshape([-1,1])
print(x)
print(y)

a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a[1:])
print(a[1:,0:1])
print(a[a<4].reshape([-1,1]))
```

### [scipy](https://docs.scipy.org/doc/scipy/reference/main_namespace.html)

```python
import numpy as np
from scipy.cluster.vq import vq, kmeans, whiten
import matplotlib.pyplot as plt
features  = np.array([[ 1.9,2.3],
                      [ 1.5,2.5],
                      [ 0.8,0.6],
                      [ 0.4,1.8],
                      [ 0.1,0.1],
                      [ 0.2,1.8],
                      [ 2.0,0.5],
                      [ 0.3,1.5],
                      [ 1.0,1.0]])
whitened = whiten(features)
book = np.array((whitened[0],whitened[2]))
kmeans(whitened,book)
```

```python
import numpy as np
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse import csr_matrix

arr = np.array([[0, 1, 0, 1],
            [1, 1, 1, 1],
            [2, 1, 1, 0],
            [0, 1, 0, 1]
])

newarr = csr_matrix(arr)

print(depth_first_order(newarr, 1))
```

### networkx

```python
# 导入库
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 创建一个邻接矩阵
adjacency_matrix = np.array([[0, 1, 1, 1],
                             [1, 0, 1, 0],
                             [1, 1, 0, 1],
                             [1, 0, 1, 0]])

# 将邻接矩阵转换为图形对象
G = nx.from_numpy_array(adjacency_matrix)

# 使用matplotlib绘制图形
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.show()

```

