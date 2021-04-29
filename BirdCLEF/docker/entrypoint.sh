#!/bin/bash

function run_as_user() {
    if [ $# -eq 1 ]; then
        PROJECT_ROOT=$1
        cd ${PROJECT_ROOT} && pip install -e .
        cd ~
    fi
    KAGGLE_ROOT=$(dirname ${PROJECT_ROOT})
    if [ -e ${KAGGLE_ROOT}/kaggle.json ]; then
        if [ ! -e ${HOME}/.kaggle/kaggle.json ]; then
            mkdir ${HOME}/.kaggle
            cp ${KAGGLE_ROOT}/kaggle.json ${HOME}/.kaggle/
        fi
    fi
    jupyter lab --ip='*' --port=8888 --no-browser --NotebookApp.token='' --notebook-dir=/home/`id -n -u`/work
}

/usr/sbin/sshd
RUN_AS_USER=$(declare -f run_as_user)
PROJECT_DOCKER_DIR=$(dirname $0)
PROJECT_ROOT=$(dirname ${PROJECT_DOCKER_DIR})
sudo LD_LIBRARY_PATH=${LD_LIBRARY_PATH} -H -u ${USER_NAME} bash -c "${RUN_AS_USER}"; run_as_user ${PROJECT_ROOT}
