#!/usr/bin/env bash

# Multi-GPU training
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS /path_to_maskrcnn_benchmark/tools/train_net.py
--config-file path/to/config/file.yaml

python -m torch.distributed.launch --nproc_per_node=8 /path_to_maskrcnn_benchmark/tools/train_net.py
--config-file path/to/config/file.yaml