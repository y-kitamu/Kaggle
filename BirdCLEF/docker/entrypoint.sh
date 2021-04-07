#!/bin/bash

function run_jupyter() {
    if [ $# -eq 1 ]; then
        PROJECT_ROOT=$1
        cd ${PROJECT_ROOT} && pip install -e .
        cd ~
    fi
    jupyter lab --ip='*' --port=8888 --no-browser --NotebookApp.token='' --notebook-dir=/home/`id -n -u`/work
}

/usr/sbin/sshd
RUN_JUPYTER=$(declare -f run_jupyter)
PROJECT_DOCKER_DIR=$(dirname $0)
PROJECT_ROOT=$(dirname ${PROJECT_DOCKER_DIR})
sudo LD_LIBRARY_PATH=${LD_LIBRARY_PATH} -H -u ${USER_NAME} bash -c "${RUN_JUPYTER}; run_jupyter ${PROJECT_ROOT}"
