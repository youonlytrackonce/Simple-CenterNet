import os
import copy
import cv2
import numpy as np
import torch
import torchvision
import transforms
import detection.transforms
from torch.utils.data import Dataset, DataLoader


class DetectionDataset(Dataset):        
    def __init__(self,
                 dataset="custom",
                 dataset_root_path="./dataset/train",
                 img_size=(1088, 608),
                 resize_method="letter_box",
                 stride=4,
                 bboxes_regression_method="wh",
                 num_classes=20,
                 augment=False,
                 normalize_transform=None):
        
        img_exts = [".png", ".jpg", ".bmp"]
        label_exts = [".txt"]

        self.dataset = dataset
        self.resize_method = resize_method  # "letter_box" or "base"
        self.img_size = img_size
        self.stride = stride  # one of [1, 2, 4, 8 ...]
        self.gt_size = (img_size[0] // self.stride, img_size[1] // self.stride)  # (w//self.stride, h//self.stride)
        self.bboxes_regression_method = bboxes_regression_method  # "ltrb" or "wh"
        self.num_classes = num_classes
        
        if self.dataset == "custom":
            files = sorted(os.listdir(dataset_root_path))
            imgs_path = [os.path.join(dataset_root_path, file).replace(os.sep, "/") for file in files if file.lower().endswith(tuple(img_exts))]
            labels_path = [os.path.join(dataset_root_path, file).replace(os.sep, "/") for file in files if file.lower().endswith(tuple(label_exts))]
        elif self.dataset == "MOT":
            pass
        
        self.imgs_path = imgs_path
        self.labels = [np.loadtxt(label_path,
                                  dtype=np.float32,
                                  delimiter=' ').reshape(-1, 5) 
                       for label_path in labels_path]
        
    def __getitem__(self, idx):
        img = cv2.imread(self.imgs_path[idx])
        label = copy.deepcopy(self.labels[idx])

        class_ids = label[:, 0]
        class_ids = class_ids.astype(np.long)

        bboxes = label[:, 1:]  # normalized xywh

        if self.resize_method == "base":  #aspect ratio is not preserved.
            img = cv2.resize(img, dsize=self.img_size)
        elif self.resize_method == "letter_box":  #aspect ratio is preserved.
            img, class_ids, bboxes = detection.transforms.letter_box_resize(img=img,
                                                 dsize=self.img_size,
                                                 class_ids=class_ids,
                                                 bboxes=bboxes)
        
        gt_heatmap, gt_positive_mask, gt_offset, gt_bboxes = detection.transforms.build_gt_tensors(img_size=self.img_size, 
                                                                                                gt_size=self.gt_size, 
                                                                                                bboxes_regression_method=self.bboxes_regression_method,
                                                                                                num_classes=self.num_classes, 
                                                                                                class_ids=class_ids, 
                                                                                                bboxes=bboxes)
        
    def __len__(self):
        return len(self.imgs_path)
