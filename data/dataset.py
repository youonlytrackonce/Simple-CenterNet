from . import voc0712
from . import coco17
from . import transforms

import cv2
import numpy as np
import torch 
from torch.utils.data import Dataset


class DetectionDataset(Dataset):  # for training/testing
    def __init__(self,
                 root,
                 dataset_name,
                 set,
                 num_classes=20,
                 img_w=512, img_h=512, stride=4,
                 use_augmentation=False,
                 keep_ratio=False):
        
        assert set == "train" or set == "test"
        assert dataset_name in ["voc", "coco", "custom"]
        
        if dataset_name == "voc":
            if set == "train":
                image_sets=[("2007", "trainval"), ("2012", "trainval")]
            elif set == "test":
                image_sets=[("2007", "test")]
                
            keep_difficult=False
            self.dataset = voc0712.VOCDetection(root, 
                                                image_sets, 
                                                keep_difficult=keep_difficult)
        elif dataset_name == "coco":
            if set == "train":
                image_set = "train2017"
            elif set == "test":
                image_set = "val2017"
            self.dataset = coco17.COCODetection(root, 
                                                image_set)
        elif dataset_name == "custom":
            pass
        
        self.num_classes = num_classes
        
        self.img_w = img_w
        self.img_h = img_h
        self.stride = stride
        
        self.heatmap_w = img_w // stride
        self.heatmap_h = img_h // stride
        
        self.keep_ratio = keep_ratio
        self.use_augmentation = use_augmentation
        
        
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        org_img_shape = [img.shape[1], img.shape[0]]
        padded_ltrb = [0, 0, 0, 0]
        
        bboxes_class = label[:, 0].reshape(-1, 1)
        bboxes_cxcywh = label[:, 1:].reshape(-1, 4)

        #resize image
        if self.keep_ratio:
            img, bboxes_cxcywh, org_img_shape, padded_ltrb  = transforms.aspect_ratio_preserved_resize(img,
                                                                                                       dsize=(self.img_w, self.img_h),
                                                                                                       bboxes_cxcywh=bboxes_cxcywh)
        else:        
            img = cv2.resize(img, dsize=(self.img_w, self.img_h))
        
        #augmentation
        if self.use_augmentation:
            img, bboxes_cxcywh, bboxes_class = transforms.random_crop(img, bboxes_cxcywh, bboxes_class, p=1.0)
            img, bboxes_cxcywh, bboxes_class = transforms.mosaic(img, bboxes_cxcywh, bboxes_class, self.dataset, self.keep_ratio, p=0.5)
            img, bboxes_cxcywh, bboxes_class = transforms.mixup(img, bboxes_cxcywh, bboxes_class, self.dataset, self.keep_ratio, use_mosaic=True, p=0.5, mosaic_p=0.5)
            img, bboxes_cxcywh = transforms.horizontal_flip(img, bboxes_cxcywh, p=0.5)
            img, bboxes_cxcywh = transforms.random_translation(img, bboxes_cxcywh, p=1.0)
            img, bboxes_cxcywh = transforms.random_scale(img, bboxes_cxcywh, p=1.0)
            img = transforms.augment_hsv(img)

        #numpy(=opencv)img 2 pytorch tensor        
        img = img[..., ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.tensor(img, dtype=torch.float32)/255.
        
        label = np.concatenate([bboxes_class, bboxes_cxcywh], axis=1)
        annotations = label.copy()
        
        label[:, 1:] = np.clip(label[:, 1:], a_min=0., a_max=1.)

        label[:, [1, 3]] *= self.heatmap_w
        label[:, [2, 4]] *= self.heatmap_h
        
        label = label[ (label[:, 3] * self.stride >= 3) & (label[:, 4] * self.stride >= 3) ] # size filtering
        
        bboxes_regression = np.zeros(shape=(4, self.heatmap_h, self.heatmap_w), dtype=np.float32)
        classes_gaussian_heatmap = np.zeros(shape=(self.num_classes, self.heatmap_h, self.heatmap_w), dtype=np.float32)
        foreground = np.zeros(shape=(self.heatmap_h, self.heatmap_w), dtype=np.float32)
        
        for bbox in label:
            bbox_class = int(bbox[0])
            
            bbox_fcx, bbox_fcy, bbox_w, bbox_h = bbox[1:]
            bbox_icx, bbox_icy = int(bbox_fcx), int(bbox_fcy)

            foreground[bbox_icy, bbox_icx] = 1
            
            bboxes_regression[0, bbox_icy, bbox_icx] = bbox_fcx-bbox_icx
            bboxes_regression[1, bbox_icy, bbox_icx] = bbox_fcy-bbox_icy
            bboxes_regression[2, bbox_icy, bbox_icx] = bbox_w
            bboxes_regression[3, bbox_icy, bbox_icx] = bbox_h
            
            classes_gaussian_heatmap[bbox_class] = transforms.scatter_gaussian_kernel(classes_gaussian_heatmap[bbox_class], bbox_icx, bbox_icy, bbox_w.item(), bbox_h.item())
        
        annotations = torch.tensor(annotations)
        bboxes_regression = torch.tensor(bboxes_regression)
        classes_gaussian_heatmap = torch.tensor(classes_gaussian_heatmap)
        foreground = torch.tensor(foreground)

        data = {}
        data["img"] = img
        data["label"] = {"annotations": annotations,"bboxes_regression": bboxes_regression, "classes_gaussian_heatmap": classes_gaussian_heatmap, "foreground": foreground}
        data["idx"] = idx
        data["org_img_shape"] = org_img_shape
        data["padded_ltrb"] = padded_ltrb
        
        return data
    
    def __len__(self):
        return len(self.dataset)
    
def collate_fn(batch_data):

    batch_img = []
    batch_annotations = []
    batch_bboxes_regression = []
    batch_classes_gaussian_heatmap = []
    batch_foreground = []
    batch_idx = []
    batch_org_img_shape = []
    batch_padded_ltrb = []

    for data in batch_data:
        batch_img.append(data["img"])
        batch_annotations.append(data["label"]["annotations"])
        batch_bboxes_regression.append(data["label"]["bboxes_regression"])
        batch_classes_gaussian_heatmap.append(data["label"]["classes_gaussian_heatmap"])
        batch_foreground.append(data["label"]["foreground"])
        batch_idx.append(data["idx"])
        batch_org_img_shape.append(data["org_img_shape"])
        batch_padded_ltrb.append(data["padded_ltrb"])

    batch_img = torch.stack(batch_img, 0)
    batch_bboxes_regression = torch.stack(batch_bboxes_regression, 0)
    batch_classes_gaussian_heatmap = torch.stack(batch_classes_gaussian_heatmap, 0)
    batch_foreground = torch.stack(batch_foreground, 0)
    
    batch_data = {}
    batch_data["img"] = batch_img
    batch_data["label"] = {"annotations": batch_annotations, "bboxes_regression": batch_bboxes_regression, "classes_gaussian_heatmap": batch_classes_gaussian_heatmap, "foreground": batch_foreground}
    batch_data["idx"] = batch_idx
    batch_data["org_img_shape"] = batch_org_img_shape
    batch_data["padded_ltrb"] = batch_padded_ltrb

    return batch_data
