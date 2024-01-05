#/bin/bash

apt-get update
apt-get upgrade

echo "更新完成"
# 需要安装ssl
apt install libssl-dev
apt install libffi-dev

echo "开始下载Python3.7"
wget https://www.python.org/ftp/python/3.7.11/Python-3.7.11.tgz
tar -zxf Python-3.7.11.tgz

echo "安装Python3.7.11"
cd ./Python-3.7.11
./configure
make && make install

echo "创建新环境"
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
