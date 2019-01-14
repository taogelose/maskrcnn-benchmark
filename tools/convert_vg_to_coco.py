import numpy as np
from tqdm import tqdm
import json, os
from dataloaders.visual_genome import VG, VGDataLoader
from config import ModelConfig
from PIL import Image
import shutil

conf = ModelConfig()
train, val, test = VG.splits(num_val_im=conf.val_size)

# train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
#                                                batch_size=conf.batch_size,
#                                                num_workers=conf.num_workers,
#                                                num_gpus=conf.num_gpus)
result = {
    "info": {"description": "VG dataset."},
}
phase = 'train'
out_json = '/mnt/data1/zdf/data/relation/vg/stanford_filered/' + phase + '.json'
images_info = []
labels_info = []
anno_id = 0

result['categories'] = []
# exclude background for maskrcnn-benchmark!
for ind, cls_name in enumerate(train.ind_to_classes):
    if ind == 0:
        continue
    cur_dict = {
        'name': cls_name,
        'id': ind
    }
    result['categories'].append(cur_dict)

def go(cur_ind, VG):
    """go for each image"""
    cur_entry = {
        'classes': VG.gt_classes[cur_ind].copy(),
        'relations': np.unique(VG.relationships[cur_ind].copy(), axis=0),
        'boxes': VG.gt_boxes[cur_ind].copy()
    }
    cur_filename = VG.filenames[cur_ind]
    cur_img = Image.open(cur_filename)
    cur_h = cur_img.height
    cur_w = cur_img.width
    # 把它变成int！！！！
    cur_id = int(cur_filename.split('/')[-1].split('.')[0])
    images_info.append(
        {
            "file_name": cur_filename.split('/')[-1],
            "height": cur_h,
            "width": cur_w,
            "id": cur_id
        }
    )
    num_obj = 0
    # go for each obj
    for i in range(len(cur_entry['classes'])):
        num_obj += 1
        global anno_id
        x1, y1, x2, y2 = cur_entry['boxes'][i]
        labels_info.append(
            {
                "segmentation": [],  # poly
                "area": (x2-x1) * (y2-y1),  # segmentation area
                # "iscrowd": 0,
                "image_id": cur_id,
                "bbox": [x1, y1, x2-x1, y2-y1],
                "category_id": cur_entry['classes'][i],
                "id": anno_id
            },
        )

        anno_id += 1
    if num_obj == 0:
        print('fuck!!!')


result['images'] = images_info
result['annotations'] = labels_info

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def split_VG():
    for i in tqdm(range(len(test.filenames))):
        cur_f = test.filenames[i]
        src = cur_f
        dst = '/mnt/data1/zdf/data/relation/vg/stanford_filered/VG_test'
        shutil.copy(src, dst)


if __name__ == '__main__':
    for i in tqdm(range(len(train.filenames))):
        go(i, train)

    with open(out_json, 'w') as f:
        json.dump(result, f, cls=NumpyEncoder)


