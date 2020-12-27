#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage : ./upload_weights.sh <experiment title(dirname)>"
fi

CUR_DIR=$(pwd)
cd $(dirname $0)

echo "Start uploading model weights in results/${1}"

rm -rf results/upload/*.hdf5
cp results/${1}/best_val*.hdf5 results/upload/

KAGGLE_CONFIG_DIR=../ kaggle datasets version -m "${1}" -p results/upload && echo "Successfully finish upload model weight!"

cd ${CUR_DIR}
