"""VOC Dataset Classes

Original author: Yonghye Kwon
https://github.com/developer0hye
"""

import cv2
import numpy as np

import os
from pycocotools.coco import COCO

PAPER_91CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane',
'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
'empty', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'empty',
'backpack', 'umbrella', 'empty', 'empty', 'handbag', 'tie', 'suitcase',
'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
'empty', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
'cake', 'chair', 'couch', 'potted plant', 'bed', 'empty', 'dining table', 'empty',
'empty', 'toilet', 'empty', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'empty', 'book', 'clock', 'vase',
'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'empty')

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
           'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
           'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
           'kite', 'baseball bat', 'baseball glove', 'skateboard',
           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
           'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
           'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
           'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
           'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
           'refrigerator', 'book', 'clock', 'vase', 'scissors',
           'teddy bear', 'hair drier', 'toothbrush') # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/

RELEASED_80CLASSES = CLASSES

# len(CLASSES) = 80
# 원래 coco paper(Microsoft COCO: Common Objects in Context) 에서 소개하기를 91개의 클래스가 포함돼있다 했는데
# release 된 건 80개 인듯?

class COCODetection(object):
    def __init__(self, 
                 root, 
                 image_set="train2017"):
        
        self.root = root
        self.images_path = []
        self.labels = []
    
        self.coco = COCO(os.path.join(root, "annotations", "instances_" + image_set + ".json"))
        self.image_set = image_set
        
        ids = list(self.coco.imgToAnns.keys())
        for id in ids:
            image_path, label = self.read_coco_data(id)
            self.images_path.append(image_path)
            self.labels.append(label)

        print("cache")
        
    def read_coco_data(self, id):
        image_info = self.coco.loadImgs(id)[0]
        image_path = os.path.join(self.root, self.image_set, image_info['file_name'])
        
        img_h, img_w = image_info['height'], image_info['width']
        target = self.coco.imgToAnns[id] # dictonary 를 원소로 갖는 list

        label = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox'] #xywh, xy is top left point
                class_name = PAPER_91CLASSES[obj['category_id']-1]# obj['category_id'] 1-based indexing
                class_idx = RELEASED_80CLASSES.index(class_name)

                bbox_cx = (bbox[0] + bbox[2] / 2.) / img_w
                bbox_cy = (bbox[1] + bbox[3] / 2.) / img_h
                bbox_w = bbox[2] / img_w
                bbox_h = bbox[3] / img_h
                bbox_area = bbox_w * bbox_h
                
                if bbox_area == 0:
                    continue

                label.append([class_idx, bbox_cx, bbox_cy, bbox_w, bbox_h])

        label = np.array(label).reshape(-1, 5)
        label[:, 1:] = np.clip(label[:, 1:], a_min=0., a_max=1.)
        return image_path, label
        
        
    def __getitem__(self, idx):
        img = cv2.imread(self.images_path[idx], cv2.IMREAD_COLOR)
        label = self.labels[idx].copy()
        return img, label
    
    def __len__(self):
        return len(self.labels)
