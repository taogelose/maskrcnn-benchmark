import numpy as np
import cv2
import os
import json

def mask2box(mask):
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    y1 = int(np.min(rows))  # y
    x1 = int(np.min(clos))  # x
    y2 = int(np.max(rows))
    x2 = int(np.max(clos))
    return (x1, y1, x2, y2)

result = {
    "info": {"description": "xxx dataset."},
}
phase = 'train'
root_dir = os.path.join('/home/rgh/Relation/code/baseline/data/pic/segmentation/', phase)
out_json = 'temp.json'
store_segmentation = False

images_info = []
labels_info = []
img_id = 0

for index, image_name in enumerate(os.listdir(os.path.join(root_dir, 'instance'))):
    print(index, image_name)

    instance = cv2.imread(os.path.join(root_dir, 'instance', image_name), flags=cv2.IMREAD_GRAYSCALE)
    semantic = cv2.imread(os.path.join(root_dir, 'semantic', image_name), flags=cv2.IMREAD_GRAYSCALE)
    h = instance.shape[0]
    w = instance.shape[1]
    images_info.append(
        {
            "file_name": image_name[:-4]+'.jpg',
            "height": h,
            "width": w,
            "id": index
        }
    )
    instance_max_num = instance.max()
    for instance_id in range(1, instance_max_num+1):
        instance_part = instance == instance_id
        object_pos = instance_part.nonzero()
        category_id = int(semantic[object_pos[0][0], object_pos[1][0]])
        area = int(instance_part.sum())
        x1, y1, x2, y2 = mask2box(instance_part)
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        segmentation = []
        if store_segmentation:
            mask, contours, hierarchy = cv2.findContours((instance_part * 255).astype(np.uint8), cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.flatten().tolist()
                segmentation.append(contour)
                if len(contour) > 4:
                    segmentation.append(contour)
            if len(segmentation) == 0:
                print('error')
                continue
        labels_info.append(
            {
                "segmentation": segmentation,  # poly
                "area": area,  # segmentation area
                "iscrowd": 0,
                "image_id": index,
                "bbox": [x1, y1, w, h],
                "category_id": category_id,
                "id": img_id
            },
        )
        img_id += 1
    break
result["images"] = images_info
result["annotations"] = labels_info
with open(out_json, 'w') as f:
    json.dump(result, f)



