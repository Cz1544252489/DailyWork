[toc]

## 1 基础部分

### 使用环境为ubuntu 22.04 TLS x64

```bash
apt-get update
apt-get upgrade

# 需要安装ssl
apt install libssl-dev
apt install libffi-dev
```

## 2 选择最新版本的代码

Python[安装包的网页](https://www.python.org/downloads/source/)

python 3.12.1: https://www.python.org/ftp/python/3.12.1/Python-3.12.1.tgz

python 3.11.7: https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz



```bash
wget https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz
tar -zxf Python-3.11.7.tgz

cd Python-3.11.7/
./configure
make && make install

cd
python3 -m venv tf-env
source ~/tf-env/bin/activate 

```

```bash
pip install --upgrade pip

pip list

# 安装一些包
pip install torch
pip install numpy
pip install tensorflow
```

## 2.1从开始到安装到tensorflow的代码

```
apt-get update
apt-get upgrade

# 需要安装ssl
apt install libssl-dev
apt install libffi-dev

# 下载特定版本的Python
wget https://www.python.org/ftp/python/3.11.7/Python-3.11.7.tgz
tar -zxf Python-3.11.7.tgz

# 开始编译安装Python，需要的时间比较长
cd Python-3.11.7/
./configure
make && make install

# 开辟一个新的虚拟Python环境
cd
python3 -m venv tf-env
source ~/tf-env/bin/activate 

# 更新pip
pip install --upgrade pip

pip list

# 安装一些包
pip install torch
pip install numpy
pip install tensorflow
```

考虑使用自动安装的python版本

```bash
apt-get -y update
apt-get -y upgrade 

# 需要安装ssl
apt -y install libssl-dev
apt -y install libffi-dev

python3 -V

apt -y install python3.10-venv

# 开辟一个新的虚拟Python环境
cd
python3 -m venv tf-env
source ~/tf-env/bin/activate 

# 更新pip
pip install --upgrade pip

pip list

# 安装一些包
pip install torch
pip install numpy
pip install tensorflow
pip install jupyter

# 使用jupyter
jupyter notebook password
jupyter notebook --generate-config
vim ~/.jupyter/jupyter_notebook_config.py

# 到这里一下需要手动修改
jupyter notebook --allow-root
# jupyter server --allow-root
```



## 2.2 尝试安装jupyterbook

```
# 需要安装sqlite3
cd
wget https://www.sqlite.org/2023/sqlite-tools-linux-x64-3440200.zip
unzip sqlite-tools-linux-x64-3440200.zip
cp ./sqlite3 ~/tf-env/bin/

# 开启一个jupyter服务器
jupyter server --generate-config

cd
git clone https://github.com/jupyter/jupyter_client.git
```

## 2.3 ChatGPT给出的解决方案



如果你已经在远程 Ubuntu 服务器上安装并启动了 Jupyter Notebook，但无法从本地机器进行连接，可能存在以下几个问题。以下是一些解决方案：

### 1. Jupyter Notebook 配置

Jupyter Notebook 默认仅允许本地访问。要从远程访问，你需要修改一些配置：

- 运行以下命令以生成配置文件（如果尚未生成）：
  ```bash
  jupyter notebook --generate-config
  ```
- 打开生成的配置文件（通常在 `~/.jupyter/jupyter_notebook_config.py`）。
- 找到并修改以下配置项（删除行首的 `#` 并设置值）：
  - `c.NotebookApp.ip = '0.0.0.0'`（允许任何 IP 地址访问）
  - `c.NotebookApp.open_browser = False`（不自动打开浏览器，因为你是在远程服务器上运行）
  - `c.NotebookApp.port = 8888`（或你选择的其他端口）

## 3 尝试配置BDA的环境

python3.6.14版本：https://www.python.org/ftp/python/3.6.14/Python-3.6.14.tgz

python 3.7.11版本：https://www.python.org/ftp/python/3.7.11/Python-3.7.11.tgz

```bash
wget https://www.python.org/ftp/python/3.7.11/Python-3.7.11.tgz
tar -zxf Python-3.7.11.tgz

cd ./Python-3.7.11
./configure
make && make install

cd
python3 -m venv tf-env-BDA
source ~/tf-env-BDA/bin/activate

# 升级git到最新
pip install --upgrade pip
```

tensorflow 的版本下载地址：https://www.tensorflow.org/install/pip?hl=zh-cn

python3.6(仅cpu)  https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.6.0-cp36-cp36m-manylinux2010_x86_64.whl

python3.7(仅cpu)  https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.6.0-cp37-cp37m-manylinux2010_x86_64.whl

tensorflow的pypi下载地址：https://pypi.org/project/tensorflow/#history

tensorflow 1.14.0下载地址：https://pypi.org/project/tensorflow/1.14.0/#files

```bash
cd
wget https://files.pythonhosted.org/packages/f4/28/96efba1a516cdacc2e2d6d081f699c001d414cc8ca3250e6d59ae657eb2b/tensorflow-1.14.0-cp37-cp37m-manylinux1_x86_64.whl
pip install tensorflow-1.14.0-cp37-cp37m-manylinux1_x86_64.whl

# 下载BDA代码
cd
git clone https://github.com/vis-opt-group/BDA.git

# 安装依赖包
# 其中要求tensorflow的版本为1.13.*到1.15.*
cd ~/BDA
pip install -r requirements.txt
```



下面是使用BDA代码的方法：

```bash
cd ~/BDA
cd ./test_script
python3  Data_hyper_cleaning.py

# 执行以下代码前需要下载数据集
# 暂时不去考虑下面的实现问题
python3 Few_shot.py --classes=5 --examples_train=1 --examples_test=1 --meta_batch_size=1 --alpha=0.4

```



## 4 一整段用于生成可以实现BDA算法环境的bash代码

只需要运行一下代码即可

```bash
wget https://file.cz123.top/9others/CodesFile/ToBDA.sh
bash ToBDA.sh
```

以下代码放到：https://file.cz123.top/9others/CodesFile/ToBDA.sh

```bash
apt-get update
apt-get upgrade

# 需要安装ssl
apt install libssl-dev
apt install libffi-dev

wget https://www.python.org/ftp/python/3.7.11/Python-3.7.11.tgz
tar -zxf Python-3.7.11.tgz

cd ./Python-3.7.11
./configure
make && make install

cd
python3 -m venv tf-env-BDA
source ~/tf-env-BDA/bin/activate

# 升级git到最新
pip install --upgrade pip

cd
wget https://files.pythonhosted.org/packages/f4/28/96efba1a516cdacc2e2d6d081f699c001d414cc8ca3250e6d59ae657eb2b/tensorflow-1.14.0-cp37-cp37m-manylinux1_x86_64.whl
pip install tensorflow-1.14.0-cp37-cp37m-manylinux1_x86_64.whl

# 下载BDA代码
cd
git clone https://github.com/vis-opt-group/BDA.git

# 安装依赖包
# 其中要求tensorflow的版本为1.13.*到1.15.*
cd ~/BDA
pip install -r requirements.txt

cd ~/BDA
cd ./test_script
python3  Data_hyper_cleaning.py
```

