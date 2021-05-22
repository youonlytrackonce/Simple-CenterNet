from . import voc0712
from . import transforms

import cv2
import numpy as np
import torch 
import torchvision.transforms
from torch.utils.data import Dataset


class DetectionDataset(Dataset):  # for training/testing
    def __init__(self,
                 root,
                 dataset_name,
                 set,
                 img_w=512, img_h=512,
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
            pass
        elif dataset_name == "custom":
            pass
        
        self.img_w = img_w
        self.img_h = img_h
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
            img, bboxes_cxcywh, bboxes_class = transforms.random_crop(img, bboxes_cxcywh, bboxes_class)
            img, bboxes_cxcywh, bboxes_class = transforms.mosaic(img, bboxes_cxcywh, bboxes_class, self.dataset, self.keep_ratio, p=0.5)
            img, bboxes_cxcywh = transforms.horizontal_flip(img, bboxes_cxcywh, p=0.5)
            img, bboxes_cxcywh = transforms.random_translation(img, bboxes_cxcywh, p=1.0)
            img, bboxes_cxcywh = transforms.random_scale(img, bboxes_cxcywh, p=1.0)
            #img = transforms.cutout(img)
            img = transforms.augment_hsv(img)

        #numpy(=opencv)img 2 pytorch tensor        
        img = img[..., ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.tensor(img, dtype=torch.float32)/255.
        
        label = np.concatenate([bboxes_class, bboxes_cxcywh], axis=1)
        label = torch.tensor(label, dtype=torch.float32)

        data = {}
        data["img"] = img
        data["label"] = label
        data["idx"] = idx
        data["org_img_shape"] = org_img_shape
        data["padded_ltrb"] = padded_ltrb
        
        return data
    
    def __len__(self):
        return len(self.dataset)
    
def collate_fn(batch_data):

    batch_img = []
    batch_label = []
    batch_idx = []
    batch_org_img_shape = []
    batch_padded_ltrb = []

    for data in batch_data:
        batch_img.append(data["img"])
        batch_label.append(data["label"])
        batch_idx.append(data["idx"])
        batch_org_img_shape.append(data["org_img_shape"])
        batch_padded_ltrb.append(data["padded_ltrb"])

    batch_img = torch.stack(batch_img, 0)
    
    batch_data = {}
    batch_data["img"] = batch_img
    batch_data["label"] = batch_label
    batch_data["idx"] = batch_idx
    batch_data["org_img_shape"] = batch_org_img_shape
    batch_data["padded_ltrb"] = batch_padded_ltrb

    return batch_data
