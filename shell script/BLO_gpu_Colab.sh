#/bin/bash

##########################################
#
# 用于更新运行环境以使用GPU
#
############################################

# 降级 gcc

# 更新apt
echo "deb [arch=amd64] http://archive.ubuntu.com/ubuntu focal main universe" >> /etc/apt/sources.list

# 安装
apt update
apt-get install gcc-7 g++-7

# 切换版本
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 80
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 80

# 卸载 cuda 12.2
rm -rf /usr/local/cuda-12.2
rm /usr/local/cuda
rm /usr/local/cuda-12
apt autoremove 

# 安装 cuda 10.0 (方法2)
# 该方法需要手动选择
# Do you accept the previously read EULA? accept
# (You are attempting to install on an unsupported configuration. Do you wish to continue? yes)
# Install NIVDIA Accelerated Graphics Driver for Linux-x86_64 xxx? no
# Install the CUDA 10.0 Toolkit? yes
# Enter Toolkit Location. default(enter)
# Do you want to install a symbolic link at /usr/local/cuda? yes
# Install the CUDA 10.0 Samples? no

cd
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
mv cuda_10.0.130_410.48_linux cuda_10.0.130_410.48_linux.run
sh cuda_10.0.130_410.48_linux.run
rm cuda_10.0.130_410.48_linux.run


# cuda的环境配置
echo "export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc
echo "export TF_XLA_FLAGS=--tf_xla_cpu_global_jit" >>/.bashrc
# 下面这一行需要在整个运行结束后执行一次（脚本内似乎无效）
# source ~/.bashrc


# 安装 cuDNN
# 下载链接在一个自己的VPS中，因为源网站无法不登录无法下载
wget http://104.244.90.25:12345/libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb
dpkg -i libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb

wget http://104.244.90.25:12345/libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb
dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.0_amd64.deb

wget http://104.244.90.25:12345/libcudnn7-doc_7.6.5.32-1+cuda10.0_amd64.deb
dpkg -i libcudnn7-doc_7.6.5.32-1+cuda10.0_amd64.deb

rm libcudnn*.deb

############################################
#
#	以下为原来的部分  可以直接运行，但是无法使用gpu
#
#########################################
apt-get -y update
apt-get -y upgrade

echo "更新完成"
# 需要安装ssl
apt -y install libssl-dev
apt -y install libffi-dev

# 可能存在并没有默认安装以下应用的服务器
apt -y install pip
apt -y install git

echo "开始下载Python3.7"
wget https://www.python.org/ftp/python/3.7.11/Python-3.7.11.tgz
tar -zxf Python-3.7.11.tgz

echo "安装Python3.7.11"
cd ./Python-3.7.11
./configure
make && make install

cd ~
rm Python-3.7.11.tgz

echo "创建新环境"
cd ~
python3 -m venv tf-env
source ~/tf-env/bin/activate

# 升级git到最新
pip install --upgrade pip

# 安装tensorflow-1.14版本
cd ~

wget https://files.pythonhosted.org/packages/f4/28/96efba1a516cdacc2e2d6d081f699c001d414cc8ca3250e6d59ae657eb2b/tensorflow-1.14.0-cp37-cp37m-manylinux1_x86_64.whl
pip install tensorflow-1.14.0-cp37-cp37m-manylinux1_x86_64.whl
wget https://files.pythonhosted.org/packages/32/67/559ca8408431c37ad3a17e859c8c291ea82f092354074baef482b98ffb7b/tensorflow_gpu-1.14.0-cp37-cp37m-manylinux1_x86_64.whl
pip install tensorflow_gpu-1.14.0-cp37-cp37m-manylinux1_x86_64.whl

rm tensorflow*.whl

pip install protobuf==3.20.*

# 直接安装库里的版本会出现未预知的错误
# pip install tensorflow==1.14
# pip install tensorflow-gpu==1.14

# 下载BOML代码
cd
git clone https://github.com/dut-media-lab/BOML.git

# 安装依赖包
# 其中要求tensorflow的版本为1.13.*到1.15.*
cd ~/BOML
pip install -r requirements.txt


# 整个过程可以使用以下代码解决
# 前两行用于清除来自numpy的警告
sed -i "1iimport warnings" ~/BOML/test_script/script_helper.py
sed -i "2iwarnings.filterwarnings('ignore',category=FutureWarning)" ~/BOML/test_script/script_helper.py

# 后一行用消除来自tensorflow的警告
sed -i "5itf.get_logger().setLevel('ERROR')" ~/BOML/boml/extension.py


# 下载BDA代码
cd
git clone https://github.com/Cz1544252489/BDA.git

# 安装依赖包
# 其中要求tensorflow的版本为1.13.*到1.15.*
cd ~/BDA
pip install -r requirements.txt

## 由于Cz1544252489的BDA库中已经有了以下修改，以下内容注释掉
# 整个过程可以使用以下代码解决
# 前两行用于清除来自numpy的警告
# sed -i "1iimport warnings" ~/BDA/test_script/Data_hyper_cleaning.py
# sed -i "2iwarnings.filterwarnings('ignore',category=FutureWarning)" ~/BDA/test_script/Data_hyper_cleaning.py
# 后一行用消除来自tensorflow的警告
# sed -i "5itf.get_logger().setLevel('ERROR')" ~/BDA/boml/extension.py

# 提示
echo "执行以下代码后可以正常使用: "
echo "source ~/.bashrc && source ~/tf-env/bin/activate "
