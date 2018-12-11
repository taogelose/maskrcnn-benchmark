#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1

#config='configs/HumanCollection_mini/e2e_mask_rcnn_R_50_C4_1x.yaml'
#LOG=log/e2e_mask_rcnn_R_50_C4_1x-`date +%Y-%m-%d_%H-%M-%S`.log
#echo $LOG
#python ./tools/train_net.py --config-file ${config} \
#2>&1 | tee ${LOG}


config='configs/HumanCollection_mini/e2e_mask_rcnn_R_50_FPN_1x.yaml'
LOG=log/e2e_mask_rcnn_R_50_FPN_1x-`date +%Y-%m-%d_%H-%M-%S`.log
echo $LOG
python ./tools/train_net.py --config-file ${config} \
2>&1 | tee ${LOG}