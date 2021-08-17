from .resnet import *
from .dcn import *
from utils import common

import numpy as np
import math
import torch
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
import torch.nn as nn

import torchvision.transforms.functional as F


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[:, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    
class Upsamling(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=2):
        super(Upsamling, self).__init__()
        # deconv basic config
        if ksize == 4:
            padding = 1
            output_padding = 0
        elif ksize == 3:
            padding = 1
            output_padding = 1
        elif ksize == 2:
            padding = 0
            output_padding = 0
        
        self.conv = DeformableConv2d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.up = nn.ConvTranspose2d(out_channels, out_channels, ksize, stride=stride, padding=padding, output_padding=output_padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        fill_up_weights(self.up)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv(x)))
        x = torch.relu(self.bn2(self.up(x)))
        return x
    
class CenterNet(nn.Module):
    def __init__(self,
                 num_classes=20,
                 pretrained_backbone=True):
        super(CenterNet, self).__init__()
        
        self.num_classes = num_classes
        self.stride = 4
        
        self.backbone = resnet18(pretrained=pretrained_backbone)
        
        self.upsample1 = Upsamling(512, 256, ksize=4, stride=2) # 32 -> 16
        self.upsample2 = Upsamling(256, 128, ksize=4, stride=2) # 16 -> 8
        self.upsample3 = Upsamling(128, 64, ksize=4, stride=2) #  8 -> 4

        self.cls_pred = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.num_classes, kernel_size=1)
        )
        
        init_prob = 0.01
        nn.init.constant_(self.cls_pred[-1].bias, -torch.log(torch.tensor((1.-init_prob)/init_prob)))
             
        self.txty_pred = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )
       
        self.twth_pred = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1)
        )
        
        #for decoding
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=1, padding=3)
        self.max_num_dets = 100

    def encode(self, x, flip=False):
        if flip==True:
            x = F.hflip(x)
        
        x = self.backbone(x)
        
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        
        cls_pred = self.cls_pred(x)
        txty_pred = self.txty_pred(x)
        twth_pred = self.twth_pred(x)
        
        out = torch.cat([txty_pred, twth_pred, cls_pred], dim=1)
        return out 
    
    def forward(self, x, flip=False):
        self.img_h, self.img_w = x.shape[2:]

        out = self.encode(x)

        if flip:
            flipped_out = self.encode(x, flip=True)
            out = (out + F.hflip(flipped_out))/2.
        
        if self.training:
            return out
        else:
            out_h, out_w = out.shape[2:]
            device = out.device
            
            grid_y = torch.arange(out_h, dtype=out.dtype, device=device).view(1, out_h, 1).repeat(1, 1, out_w)
            grid_x = torch.arange(out_w, dtype=out.dtype, device=device).view(1, 1, out_w).repeat(1, out_h, 1)
            
            # localization
            bboxes_cx = (self.stride * (grid_x + out[:, 0]).flatten(start_dim=1))/self.img_w
            bboxes_cy = (self.stride * (grid_y + out[:, 1]).flatten(start_dim=1))/self.img_h
            
            bboxes_w = (self.stride * out[:, 2].flatten(start_dim=1))/self.img_w
            bboxes_h = (self.stride * out[:, 3].flatten(start_dim=1))/self.img_h
            
            class_heatmap = torch.sigmoid(out[:, 4:])# [B, 20, H, W]
            class_heatmap = self.nms(class_heatmap).flatten(start_dim=2).transpose(1, 2) # [B, 20, H*W] -> # [B, H*W, 20]
            class_heatmap, class_idx = torch.max(class_heatmap, dim=2) # [B, H*W]
            
            _, topk_inds = torch.topk(class_heatmap, k=self.max_num_dets, dim=1)
            
            out = [torch.gather(x, dim=1, index=topk_inds) 
                   for x in [class_idx.float(), bboxes_cx, bboxes_cy, bboxes_w, bboxes_h, class_heatmap]]
            out = torch.stack(out, dim=2) # [B, self.max_num_dets, 6]
            return out
    
    def nms(self, class_probability:Tensor) -> Tensor:
        mask = torch.eq(class_probability, self.max_pool(class_probability)).to(class_probability.dtype)
        return class_probability * mask
    
    def post_processing(self, 
                        batch_bboxes : Tensor,
                        batch_org_img_shape,
                        batch_padded_ltrb, 
                        confidence_threshold : float=1e-2):
        self.eval()
        with torch.no_grad():
            filtered_batch_bboxes = []
            for bboxes, org_img_shape, padded_ltrb in zip(batch_bboxes, batch_org_img_shape, batch_padded_ltrb):
                filtered_bboxes = {"class": np.zeros((0, 1)), "position": np.zeros((0, 4)), "confidence": np.zeros((0, 1)), "num_detected_bboxes": 0}
                
                bboxes_confidence = bboxes[:, 5]
                confidence_mask = bboxes_confidence > confidence_threshold

                if torch.count_nonzero(confidence_mask) > 0:
                    
                    bboxes = bboxes[confidence_mask]
                    bboxes = common.reconstruct_bboxes(normalized_bboxes=bboxes,
                                                     resized_img_shape=(self.img_w, self.img_h),
                                                     padded_ltrb=padded_ltrb,
                                                     org_img_shape=org_img_shape)
                    
                    filtered_bboxes["num_detected_bboxes"] = len(bboxes)
                    filtered_bboxes["class"] = bboxes[:, 0].cpu().numpy()
                    filtered_bboxes["position"] = bboxes[:, 1:5].cpu().numpy()
                    filtered_bboxes["confidence"] = bboxes[:, 5].cpu().numpy()

                filtered_batch_bboxes.append(filtered_bboxes)

            return filtered_batch_bboxes
   
    def compute_loss(self, batch_pred, batch_label):
        batch_size = batch_pred.shape[0]
        dtype = batch_pred.dtype
        device = batch_pred.device
        
        loss_offset_xy_function = nn.L1Loss(reduction='none')
        loss_wh_function = nn.L1Loss(reduction='none')

        batch_label["bboxes_regression"] = batch_label["bboxes_regression"].to(device)
        batch_label["classes_gaussian_heatmap"] = batch_label["classes_gaussian_heatmap"].to(device)
        batch_label["foreground"] = batch_label["foreground"].to(device)
        
        batch_loss_offset_x = torch.tensor(0., dtype=dtype, device=device)
        batch_loss_offset_y = torch.tensor(0., dtype=dtype, device=device)
        batch_loss_w = torch.tensor(0., dtype=dtype, device=device)
        batch_loss_h = torch.tensor(0., dtype=dtype, device=device)
        batch_loss_class_heatmap = torch.tensor(0., dtype=dtype, device=device)
        
        batch_loss_offset_x = loss_offset_xy_function(batch_pred[:, 0], batch_label["bboxes_regression"][:, 0]) * batch_label["foreground"]/batch_size
        batch_loss_offset_y = loss_offset_xy_function(batch_pred[:, 1], batch_label["bboxes_regression"][:, 1]) * batch_label["foreground"]/batch_size
        batch_loss_w = loss_wh_function(batch_pred[:, 2], batch_label["bboxes_regression"][:, 2]) * batch_label["foreground"]/batch_size
        batch_loss_h = loss_wh_function(batch_pred[:, 3], batch_label["bboxes_regression"][:, 3]) * batch_label["foreground"]/batch_size
        
        batch_loss_class_heatmap = focal_loss(batch_pred[:, 4:], batch_label["classes_gaussian_heatmap"])/batch_size
        
        batch_loss_offset_x = batch_loss_offset_x.flatten(1).sum(1)
        batch_loss_offset_y = batch_loss_offset_y.flatten(1).sum(1)
        batch_loss_w = batch_loss_w.flatten(1).sum(1)
        batch_loss_h = batch_loss_h.flatten(1).sum(1)
        batch_loss_class_heatmap = batch_loss_class_heatmap.flatten(1).sum(1)

        batch_num_positive_samples = batch_label["foreground"].flatten(1).sum(1)
        batch_num_positive_samples = torch.maximum(batch_num_positive_samples, torch.ones_like(batch_num_positive_samples)) # to avoid zero divide
        
        batch_loss_offset_x /= batch_num_positive_samples
        batch_loss_offset_y /= batch_num_positive_samples
        batch_loss_w /= batch_num_positive_samples
        batch_loss_h /= batch_num_positive_samples
        batch_loss_class_heatmap /= batch_num_positive_samples

        batch_loss_offset_xy = torch.sum(batch_loss_offset_x + batch_loss_offset_y)/2.
        batch_loss_wh = 0.1 * torch.sum(batch_loss_w + batch_loss_h)/2.
        batch_loss_class_heatmap = torch.sum(batch_loss_class_heatmap)
        loss = batch_loss_offset_xy + batch_loss_wh + batch_loss_class_heatmap
        return loss, [batch_loss_offset_xy, batch_loss_wh, batch_loss_class_heatmap]

# Gaussan Kernels for Training Class Heatmap, read Training-Time-Friendly Network for Real-Time Object Detection paper for more details
def scatter_gaussian_kernel(heatmap, bbox_icx, bbox_icy, bbox_w, bbox_h, alpha=0.54):
    heatmap_h, heatmap_w = heatmap.shape
    dtype = heatmap.dtype
    device = heatmap.device
    
    std_w = alpha * bbox_w/6.
    std_h = alpha * bbox_h/6.
    
    var_w = std_w ** 2
    var_h = std_h ** 2
    
    grid_y, grid_x = torch.meshgrid([torch.arange(heatmap_h, dtype=dtype, device=device),
                                     torch.arange(heatmap_w, dtype=dtype, device=device)])
    
    gaussian_kernel = torch.exp(-((grid_x - bbox_icx)**2/(2. * var_w))-((grid_y - bbox_icy)**2/(2. * var_h)))
    gaussian_kernel[bbox_icy, bbox_icx] = 1.
    heatmap = torch.maximum(heatmap, gaussian_kernel)
    return heatmap

#Read Training-Time-Friendly Network for Real-Time Object Detection paper for more details
def focal_loss(pred, gaussian_kernel, alpha=2., beta=4., eps=1e-5):
    pred = torch.sigmoid(pred).clamp(eps, 1.-eps)
    
    positive_mask = gaussian_kernel == 1.
    negative_mask = ~positive_mask

    positive_loss = -(((1. - pred) ** alpha) * torch.log(pred)) * positive_mask.float()
    negative_loss = -(((1. - gaussian_kernel) ** beta) * (pred ** alpha) * torch.log(1.-pred)) * negative_mask.float()
    
    if torch.count_nonzero(positive_mask) == 0:
        return negative_loss
    elif torch.count_nonzero(negative_mask) == 0:
        return positive_loss
    else:
        return negative_loss + positive_loss
