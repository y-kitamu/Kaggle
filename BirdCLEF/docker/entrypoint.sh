#!/bin/bash

function run_jupyter() {
    jupyter lab --ip='*' --port=8888 --no-browser --allow-root --NotebookApp.token='' --notebook-dir=/home/`id -n -u`/work
}

/usr/sbin/sshd
sudo -u ${USER_NAME} -sH run_jupyter
