#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
#PORT=${PORT:-29500}
PORT=${PORT:-29508}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_moco.py $CONFIG --launcher pytorch ${@:3}