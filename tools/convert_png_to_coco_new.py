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
    "info": {"description": "PIC v1 dataset."},
}
phase = 'val'
root_dir = os.path.join('/mnt/data3/DataSet/PIC/segmentation/', phase)
out_json = phase +'.json'
store_segmentation = False
error_list = []
for line in open('val_error_list.txt', 'r'):
    error_list.append(line[:-1])
images_info = []
labels_info = []
anno_id = 0
cnt = 0
for index, image_name in enumerate(os.listdir(os.path.join(root_dir, 'instance'))):
    if image_name in error_list:
        continue
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
            "id": cnt
        }
    )
    instance_max_num = instance.max()
    for instance_id in range(1, instance_max_num+1):
        instance_part = instance == instance_id
        category_id = int((instance_part * semantic).max())
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
                #segmentation.append(contour)
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
                "image_id": cnt,
                "bbox": [x1, y1, w, h],
                "category_id": category_id,
                "id": anno_id
            },
        )
        anno_id += 1
    cnt += 1
    #break

result["images"] = images_info
result["annotations"] = labels_info
result["categories"] = [
        {
            "name": "human",
            "id": 1
        },
        {
            "name": "floor",
            "id": 2
        },
        {
            "name": "bed",
            "id": 3
        },
        {
            "name": "window",
            "id": 4
        },
        {
            "name": "cabinet",
            "id": 5
        },
        {
            "name": "door",
            "id": 6
        },
        {
            "name": "table",
            "id": 7
        },
        {
            "name": "plant",
            "id": 8
        },
        {
            "name": "curtain",
            "id": 9
        },
        {
            "name": "chair",
            "id": 10
        },
        {
            "name": "sofa",
            "id": 11
        },
        {
            "name": "shelf",
            "id": 12
        },
        {
            "name": "rug",
            "id": 13
        },
        {
            "name": "lamp",
            "id": 14
        },
        {
            "name": "fridge",
            "id": 15
        },
        {
            "name": "stairs",
            "id": 16
        },
        {
            "name": "pillow",
            "id": 17
        },
        {
            "name": "kitchen",
            "id": 18
        },
        {
            "name": "sculpture",
            "id": 19
        },
        {
            "name": "sink",
            "id": 20
        },
        {
            "name": "document",
            "id": 21
        },
        {
            "name": "painting",
            "id": 22
        },
        {
            "name": "barrel",
            "id": 23
        },
        {
            "name": "basket",
            "id": 24
        },
        {
            "name": "poke",
            "id": 25
        },
        {
            "name": "stool",
            "id": 26
        },
        {
            "name": "clothes",
            "id": 27
        },
        {
            "name": "bottle",
            "id": 28
        },
        {
            "name": "plate",
            "id": 29
        },
        {
            "name": "cellphone",
            "id": 30
        },
        {
            "name": "toy",
            "id": 31
        },
        {
            "name": "cushion",
            "id": 32
        },
        {
            "name": "box",
            "id": 33
        },
        {
            "name": "display",
            "id": 34
        },
        {
            "name": "blanket",
            "id": 35
        },
        {
            "name": "pot",
            "id": 36
        },
        {
            "name": "nameplate",
            "id": 37
        },
        {
            "name": "banners",
            "id": 38
        },
        {
            "name": "cup",
            "id": 39
        },
        {
            "name": "pen",
            "id": 40
        },
        {
            "name": "digital",
            "id": 41
        },
        {
            "name": "cooker",
            "id": 42
        },
        {
            "name": "umbrella",
            "id": 43
        },
        {
            "name": "decoration",
            "id": 44
        },
        {
            "name": "straw",
            "id": 45
        },
        {
            "name": "certificate",
            "id": 46
        },
        {
            "name": "food",
            "id": 47
        },
        {
            "name": "club",
            "id": 48
        },
        {
            "name": "towel",
            "id": 49
        },
        {
            "name": "pet",
            "id": 50
        },
        {
            "name": "tool",
            "id": 51
        },
        {
            "name": "appliance",
            "id": 52
        },
        {
            "name": "pram",
            "id": 53
        },
        {
            "name": "car",
            "id": 54
        },
        {
            "name": "grass",
            "id": 55
        },
        {
            "name": "vegetation",
            "id": 56
        },
        {
            "name": "water",
            "id": 57
        },
        {
            "name": "ground",
            "id": 58
        },
        {
            "name": "road",
            "id": 59
        },
        {
            "name": "streetlight",
            "id": 60
        },
        {
            "name": "railing",
            "id": 61
        },
        {
            "name": "stand",
            "id": 62
        },
        {
            "name": "steps",
            "id": 63
        },
        {
            "name": "pillar",
            "id": 64
        },
        {
            "name": "awnings",
            "id": 65
        },
        {
            "name": "building",
            "id": 66
        },
        {
            "name": "hill",
            "id": 67
        },
        {
            "name": "stone",
            "id": 68
        },
        {
            "name": "bridge",
            "id": 69
        },
        {
            "name": "bicycle",
            "id": 70
        },
        {
            "name": "motorcycle",
            "id": 71
        },
        {
            "name": "airplane",
            "id": 72
        },
        {
            "name": "boat",
            "id": 73
        },
        {
            "name": "balls",
            "id": 74
        },
        {
            "name": "equipment",
            "id": 75
        },
        {
            "name": "apparatus",
            "id": 76
        },
        {
            "name": "gun",
            "id": 77
        },
        {
            "name": "smoke",
            "id": 78
        },
        {
            "name": "rope",
            "id": 79
        },
        {
            "name": "facilities",
            "id": 80
        },
        {
            "name": "prop",
            "id": 81
        },
        {
            "name": "armament",
            "id": 82
        },
        {
            "name": "bag",
            "id": 83
        },
        {
            "name": "instruments",
            "id": 84
        }
        ]
with open(out_json, 'w') as f:
    json.dump(result, f)


