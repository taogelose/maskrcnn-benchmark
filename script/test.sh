#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

#config='configs/HumanCollection_mini/e2e_mask_rcnn_R_50_FPN_1x.yaml'
#
#python ./tools/test_net.py --config-file ${config} \
#MODEL.WEIGHT checkpoints/fpn/model_0270000.pth

config='configs/PIC/e2e_mask_rcnn_R_101_FPN_1x.yaml'

python tools/test_net.py --config-file ${config} \
MODEL.WEIGHT checkpoints/fpn/model_0245000.pth