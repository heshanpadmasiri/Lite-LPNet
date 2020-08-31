#!/bin/zsh
DATA_PATH=$1
MODEL_NAME=$2

for idx in {0..6}
    do
        python train_lp.py "${DATA_PATH}" "${MODEL_NAME}" "${idx}"
    done

