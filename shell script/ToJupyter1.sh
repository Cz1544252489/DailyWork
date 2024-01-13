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
wget https://file.cz123.top/9others/CodesFile/requirements.txt
pip install -r requirements.txt
pip install jupyter
jupyter notebook --generate-config
sed -i '/# c.ServerApp.ip =/{s/localhost/0.0.0.0/; s/^..//;}' ~/.jupyter/jupyter_notebook_config.py
sed -i '/# c.ServerApp.open_browser/s/^..//' ~/.jupyter/jupyter_notebook_config.py
sed -i '/# c.ServerApp.port =/{s/0/8888/; s/^..//;}' ~/.jupyter/jupyter_notebook_config.py
# sed -i "/# c.ServerApp.password =/{s/''/'1544252489'/; s/^..//;}" ~/.jupyter/jupyter_notebook_config.py
apt install nginx -y
ufw allow 80
nohup jupyter notebook --allow-root > jupyter.log
token=$(sed -n '/token/p' jupyter.log | head -1 | cut -d'=' -f2-)
rm /var/www/nginx/index*
echo $token > /var/www/nginx/index.html
