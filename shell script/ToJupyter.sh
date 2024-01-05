apt-get -y update
apt-get -y upgrade
apt -y install libssl-dev
apt -y install libffi-dev
ufw allow 8888
pa=$(python3 -V | cut -c 8-11)
paname='python'$pa'-venv'
apt -y install $paname
cd
python3 -m venv tf-env
source ~/tf-env/bin/activate 
pip install torch
pip install numpy
pip install jupyter
jupyter notebook --generate-config
sed -i '/# c.ServerApp.ip =/{s/localhost/0.0.0.0/; s/^..//;}' ~/.jupyter/jupyter_notebook_config.py
sed -i '/# c.ServerApp.open_browser/s/^..//' ~/.jupyter/jupyter_notebook_config.py
sed -i '/# c.ServerApp.port =/{s/0/8888/; s/^..//;}' ~/.jupyter/jupyter_notebook_config.py
# sed -i "/# c.ServerApp.password =/{s/''/'1544252489'/; s/^..//;}" ~/.jupyter/jupyter_notebook_config.py
jupyter notebook --allow-root
