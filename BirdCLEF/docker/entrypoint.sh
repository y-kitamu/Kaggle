#!/bin/bash

function run_jupyter() {
    jupyter lab --ip='*' --port=8888 --no-browser --NotebookApp.token='' --notebook-dir=/home/`id -n -u`/work
}

/usr/sbin/sshd
RUN_JUPYTER=$(declare -f run_jupyter)
sudo LD_LIBRARY_PATH=${LD_LIBRARY_PATH} -H -u ${USER_NAME} bash -c "${RUN_JUPYTER}; run_jupyter"
