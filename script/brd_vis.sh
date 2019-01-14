#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

#config='configs/HumanCollection_mini/e2e_mask_rcnn_R_50_FPN_1x.yaml'
#
#python tools/brd_vis.py --config-file ${config} \
#MODEL.WEIGHT checkpoints/fpn/model_0250000.pth


config='configs/PIC/e2e_mask_rcnn_R_101_FPN_1x.yaml'

python tools/brd_vis.py --config-file ${config} \
MODEL.WEIGHT checkpoints/fpn/model_0112500.pth
