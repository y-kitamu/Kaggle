FROM tensorflow/tensorflow:2.4.1-gpu-jupyter

ENV DEBIAN_FRONTEND=noninteractive

ARG user
RUN useradd -m ${user} &&\
    usermod -aG sudo ${user}

# ssh (https://docs.docker.com/engine/examples/running_ssh_service/)
RUN apt-get update && apt-get upgrade -y && apt-get install openssh-server -y
RUN mkdir /var/run/sshd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

RUN apt-get update && apt-get upgrade -y &&\
    apt-get install sudo git nodejs npm libsm6 libxext6 libxrender-dev libgl1-mesa-dev -y

RUN npm install n -g && n stable
RUN apt purge -y nodejs npm

# jupyter lab
RUN python3 -m pip install --upgrade pip
RUN pip3 install jupyterlab==2.2.0 widgetsnbextension
RUN jupyter serverextension enable --py jupyterlab --sys-prefix
# lsp
RUN pip3 install jupyter-lsp
RUN jupyter labextension install @krassowski/jupyterlab-lsp  # for JupyterLab 2.x
RUN pip3 install python-language-server[all]
# variable inspector
RUN jupyter labextension install @lckr/jupyterlab_variableinspector
# tensorboard
RUN pip3 install jupyter-tensorboard
RUN jupyter labextension install jupyterlab_tensorboard
# emacs keybind
RUN jupyter labextension install jupyterlab-emacskeys

RUN jupyter lab build

ADD .style.yapf /root/
ADD .style.yapf /home/${user}/

RUN npm install pyright -g

# python packages
RUN pip3 install -U pip
ADD requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && rm -rf /tmp/requirements.txt

ADD ./protoc/bin/protoc /usr/local/bin/
RUN chmod +x /usr/local/bin/protoc
