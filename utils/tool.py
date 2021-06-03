import torch
import numpy as np
import random
import os
import shutil

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def mkdir(dir, remove_existing_dir=False):
    if os.path.isdir(dir) and remove_existing_dir:
        shutil.rmtree(dir)
    os.makedirs(dir)

def reconstruct_bboxes(normalized_bboxes, resized_img_shape, padded_ltrb, org_img_shape):
    normalized_bboxes[:, [1, 3]] *= resized_img_shape[0]
    normalized_bboxes[:, [2, 4]] *= resized_img_shape[1]
                        
    normalized_bboxes[:, 1] -= padded_ltrb[0]
    normalized_bboxes[:, 2] -= padded_ltrb[1]
    
    non_padded_img_shape = [resized_img_shape[0] - padded_ltrb[0] - padded_ltrb[2], 
                            resized_img_shape[1] - padded_ltrb[1] - padded_ltrb[3]]
    
    normalized_bboxes[:, [1, 3]] /= non_padded_img_shape[0]
    normalized_bboxes[:, [2, 4]] /= non_padded_img_shape[1]
    
    normalized_bboxes[:, [1, 3]] *= org_img_shape[0]
    normalized_bboxes[:, [2, 4]] *= org_img_shape[1]
    
    normalized_bboxes[:, [1, 3]] = torch.clamp(normalized_bboxes[:, [1, 3]], 0, org_img_shape[0])
    normalized_bboxes[:, [2, 4]] = torch.clamp(normalized_bboxes[:, [2, 4]], 0, org_img_shape[1])
    return normalized_bboxes
