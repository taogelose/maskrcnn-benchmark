from maskrcnn_benchmark.config import cfg
from demo.predictor import COCODemo
import cv2
import os
import torch
from torch import nn
import argparse


parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument(
    "--config-file",
    default="",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)


# config_file = "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml"
# config_file = "configs/e2e_mask_rcnn_R_18_FPN_1x.yaml"

# update the config options with the config file
# cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

# torch.cuda.set_device(args.local_rank)
# task_name = "fpn"
# iteration = 150000
# cfg.MODEL.WEIGHT="checkpoints/"+task_name+"/model_{}_{:07d}.pth".format(task_name, iteration)

cfg.freeze()

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
print(coco_demo.model)
# load image and then run prediction

# image = 'image/3BR_IMG_20180320_135032.jpg'
# img = cv2.imread(image)
# predictions = coco_demo.run_on_opencv_image(img)
# cv2.imwrite('predction.png', predictions)


# img_path = '../Dataset/HumanCollection_mini/coco/HC_mini_test/'
img_path = './viz/HW_test_set/'
pred_path = './viz/output/'
# method = '{}-{:07d}-'.format(task_name, iteration)
if os.path.exists(pred_path) == 0:
    os.makedirs(pred_path)
img_list = os.listdir(img_path)

for image_name in img_list:
    print(image_name)
    img = cv2.imread(img_path + image_name)
    # img = cv2.resize(img,None,fx=0.1, fy = 0.1)
    # predictions = coco_demo.run_on_opencv_image(img)
    predictions = coco_demo.brd_run_on_opencv_image(img)
    cv2.imwrite(pred_path + image_name, predictions)
