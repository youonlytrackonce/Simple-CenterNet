import torch
import numpy as np
import cv2
import random
import os
import shutil
import yaml

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def load_only_model_weights(model, weights_path, device):
    if weights_path is not None:
        from collections import OrderedDict
        chkpt = torch.load(weights_path, map_location=device)
        
        keys = chkpt['model_state_dict'].keys()
        values = chkpt['model_state_dict'].values()
        
        new_keys = []
        
        for key in keys:
            new_key = key if not 'module' in key else key[7:]
            new_keys.append(new_key)
        
        new_dict = OrderedDict(list(zip(new_keys, values)))
        model.load_state_dict(new_dict)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def parse_yaml(file_path):
    with open(file_path) as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        return yaml_data

def write_bboxes(file, img, bboxes, classes_table, draw_rect=False):
    with open(file, "w") as f:
        for bbox in bboxes:
            c = int(bbox[0])
            l = (bbox[1] - bbox[3] / 2.)
            r = (bbox[1] + bbox[3] / 2.)

            t = (bbox[2] - bbox[4] / 2.)
            b = (bbox[2] + bbox[4] / 2.)
            
            if len(bbox) == 5:                
                f.write(f"{classes_table[c]} {l} {t} {r} {b}\n")
            else:
                confidence = bbox[5]
                f.write(f"{classes_table[c]} {confidence} {l} {t} {r} {b}\n")
            
            if draw_rect:
                cv2.rectangle(img=img, pt1=(int(l), int(t)), pt2=(int(r), int(b)), color=(255, 0, 0), thickness=3)

def mkdir(dir, remove_existing_dir=False):
    if os.path.isdir(dir):
        if remove_existing_dir:
            shutil.rmtree(dir)
            os.makedirs(dir)
    else:
        os.makedirs(dir)

def get_iterations_per_epoch(training_set, batch_size):
    return len(training_set) // batch_size
    
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
