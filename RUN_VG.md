# 在maskrcnn-benchmark上跑VG数据库

在maskrcnn-benchmark, PyTorch版本1.0上跑VG数据库。

### 1.把VG的VG-SGG.h5文件里的bbox和类别之类的信息转成coco格式

- 见tools/convert_vg_to_coco
- 这里需要特别注意：VG中的train.ind_to_classes，也就是类别信息中，包含了背景类：(0: "__back_ground__")，而maskrcnn中会自动添加这一类，故转化时不应该加上背景类
- 类别的编号从1开始，maskrcnn会自动加上背景类0
- 数据库的名字写在path_catalog里，为'coco_VG_train'和'coco_VG_val'，需要加coco前缀让代码识别这是coco格式

### 2. 修改COCODataset实现

COCODataset是当前要跑的数据库的实现类 (位于coco.py)，这里没有新建一个VGDataset类，而是直接在COCODataset上进行修改。

- 从json读入bbox和类别信息后，对应json里的东西进行修改，如删除mask信息，忽略iscrowd等
- **把bbox进行scale至原图级别！** 直接拿出来的bbox是长边为1024级别，而当前图片长边不是1024，则bbox *= max(img.size) / 1024。注意：剩下的就不用管了，只要bbox跟原图对应即可，其余是maskrcnn会做的东西（如把他们一同scale到设定的输入级别，一般为1333 × 1000）


### 3. 修改类别个数信息

- 在default里把_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES改成正常类别个数 + 1，加一代表maskrcnn自动加上的背景类
 

