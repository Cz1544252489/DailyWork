#/bin/bash

apt-get -y update
apt-get -y upgrade

# 可能存在并没有默认安装以下应用的服务器
apt install pip
apt install git

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

cd
rm Python-3.7.11.tgz

echo "创建新环境"
cd
python3 -m venv tf-env
source ~/tf-env/bin/activate

# 升级git到最新
pip install --upgrade pip

cd
wget https://files.pythonhosted.org/packages/f4/28/96efba1a516cdacc2e2d6d081f699c001d414cc8ca3250e6d59ae657eb2b/tensorflow-1.14.0-cp37-cp37m-manylinux1_x86_64.whl
pip install tensorflow-1.14.0-cp37-cp37m-manylinux1_x86_64.whl
rm tensorflow-1.14.0-cp37-cp37m-manylinux1_x86_64.whl

pip install protobuf==3.20.*

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

# 提示
echo "执行以下代码后可以正常使用: "
echo "source ~/tf-env/bin/activate"
