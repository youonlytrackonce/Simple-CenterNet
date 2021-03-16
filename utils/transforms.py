import cv2
import numpy as np
import random

def cxcywh2xyxy(bboxes_cxcywh: np.ndarray):
    bboxes_xyxy = bboxes_cxcywh.copy()
    bboxes_xyxy[:, 0] = bboxes_cxcywh[:, 0] - bboxes_cxcywh[:, 2] / 2.
    bboxes_xyxy[:, 1] = bboxes_cxcywh[:, 1] - bboxes_cxcywh[:, 3] / 2.
    bboxes_xyxy[:, 2] = bboxes_cxcywh[:, 0] + bboxes_cxcywh[:, 2] / 2.
    bboxes_xyxy[:, 3] = bboxes_cxcywh[:, 1] + bboxes_cxcywh[:, 3] / 2.
    return bboxes_xyxy

def xyxy2cxcywh(bboxes_xyxy: np.ndarray):
    bboxes_cxcywh = bboxes_xyxy.copy()
    bboxes_cxcywh[:, 0] = (bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2.
    bboxes_cxcywh[:, 1] = (bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) / 2.
    bboxes_cxcywh[:, 2] = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
    bboxes_cxcywh[:, 3] = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]
    return bboxes_cxcywh

def aspect_ratio_preserved_resize(img, dsize, bboxes_cxcywh=None):
    org_img_height, org_img_width = img.shape[:2]
    target_width, target_height = dsize

    ratio = min(target_width / org_img_width, target_height / org_img_height)
    
    non_padded_resized_height = round(org_img_height * ratio)
    non_padded_resized_width = round(org_img_width * ratio)

    img = cv2.resize(img, dsize=(non_padded_resized_width, non_padded_resized_height))

    pad_left = (target_width - non_padded_resized_width) // 2
    pad_right = target_width - non_padded_resized_width - pad_left
    pad_top = (target_height - non_padded_resized_height) // 2
    pad_bottom = target_height - non_padded_resized_height - pad_top

    # padding
    img = cv2.copyMakeBorder(img,
                             pad_top,
                             pad_bottom,
                             pad_left,
                             pad_right,
                             cv2.BORDER_CONSTANT,
                             value=(127, 127, 127))
    
    assert img.shape[0] == target_height and img.shape[1] == target_width
    
    if bboxes_cxcywh is not None:
        # padding으로 인한 객체 translation 보상
        bboxes_cxcywh[:, [0, 2]] *= non_padded_resized_width
        bboxes_cxcywh[:, [1, 3]] *= non_padded_resized_height

        bboxes_cxcywh[:, 0] += pad_left
        bboxes_cxcywh[:, 1] += pad_top

        bboxes_cxcywh[:, [0, 2]] /= target_width
        bboxes_cxcywh[:, [1, 3]] /= target_height

        return img, bboxes_cxcywh, [org_img_width, org_img_height], [pad_left, pad_top, pad_right, pad_bottom]
    return img

def horizontal_flip(img, bboxes_cxcywh, p=0.5):
    if random.random() < p:
        img = cv2.flip(img, 1)#1이 호리즌탈 방향 반전
        bboxes_cxcywh[:, 0] = 1. - bboxes_cxcywh[:, 0]
        return img, bboxes_cxcywh
    return img, bboxes_cxcywh

def random_translation(img, bboxes_cxcywh, p=1.0, border_value=(127, 127, 127)):
    if random.random() < p:
        img_height, img_width = img.shape[0:2]
        
        bboxes_xyxy = cxcywh2xyxy(bboxes_cxcywh)
        
        min_tx = round(img_width * np.min(bboxes_xyxy[:, 0]))
        max_tx = img_width-round(img_width * np.max(bboxes_xyxy[:, 2]))

        min_ty = round(img_height * np.min(bboxes_xyxy[:, 1]))
        max_ty = img_height-round(img_height * np.max(bboxes_xyxy[:, 3]))

        tx = random.randint(-min_tx, max_tx)
        ty = random.randint(-min_ty, max_ty)

        # translation matrix
        tm = np.float32([[1, 0, tx],
                         [0, 1, ty]])  # [1, 0, tx], [1, 0, ty]

        img = cv2.warpAffine(img, tm, (img_width, img_height), borderValue=border_value)

        bboxes_xyxy[:, [0, 2]] += (tx / img_width)
        bboxes_xyxy[:, [1, 3]] += (ty / img_height)
        bboxes_xyxy = np.clip(bboxes_xyxy, 0., 1.)

        bboxes_cxcywh = xyxy2cxcywh(bboxes_xyxy)
        return img, bboxes_cxcywh
    return img, bboxes_cxcywh
    
def random_scale(img, bboxes_cxcywh, p=1.0, border_value=(127, 127, 127), lower_bound=0.25, upper_bound=4.0):
    if random.random() < p:
        img_height, img_width = img.shape[0:2]
        
        bboxes_xyxy = cxcywh2xyxy(bboxes_cxcywh)
        
        # calculate range
        min_bbox_w = np.min(bboxes_cxcywh[:, 2]) * img_width
        min_bbox_h = np.min(bboxes_cxcywh[:, 3]) * img_height
        
        if min_bbox_w >= 32 and min_bbox_h >= 32:
            min_scale = max(32/min_bbox_w, 32/min_bbox_h)
        else:
            min_scale = 1.
            
        max_bbox_w = np.max(bboxes_cxcywh[:, 2]) * img_width
        max_bbox_h = np.max(bboxes_cxcywh[:, 3]) * img_height
        
        max_scale = max(img_width/max_bbox_w, img_height/max_bbox_h)
        
        # clip
        min_scale = max(min_scale, lower_bound)#lower bound
        max_scale = min(max_scale, upper_bound)#upper bound
        
        cx = img_width//2
        cy = img_height//2
        
        bboxes_xmin = round(img_width * np.min(bboxes_xyxy[:, 0]))
        bboxes_ymin = round(img_height * np.min(bboxes_xyxy[:, 1]))
        bboxes_xmax = round(img_width * np.max(bboxes_xyxy[:, 2]))
        bboxes_ymax = round(img_height * np.max(bboxes_xyxy[:, 3]))

        for _ in range(50):
            random_scale = random.uniform(min_scale, max_scale)
            
            #센터 기준으로 확대 혹은 축소
            tx = cx - random_scale * cx
            ty = cy - random_scale * cy
            
            transformed_bboxes_xmin = bboxes_xmin * random_scale + tx
            transformed_bboxes_ymin = bboxes_ymin * random_scale + ty
            transformed_bboxes_xmax = bboxes_xmax * random_scale + tx
            transformed_bboxes_ymax = bboxes_ymax * random_scale + ty
          
            if transformed_bboxes_xmin < 0 or transformed_bboxes_xmax >= img_width:
                continue

            if transformed_bboxes_ymin < 0 or transformed_bboxes_ymax >= img_height:
                continue
            
            # scale matrix
            sm = np.float32([[random_scale, 0, tx],
                            [0, random_scale, ty]])  # [1, 0, tx], [1, 0, ty]

            img = cv2.warpAffine(img, sm, (img_width, img_height), borderValue=border_value)

            bboxes_xyxy *= random_scale
            bboxes_xyxy[:, [0, 2]] += (tx / img_width)
            bboxes_xyxy[:, [1, 3]] += (ty / img_height)
            bboxes_xyxy = np.clip(bboxes_xyxy, 0., 1.)
            
            bboxes_cxcywh = xyxy2cxcywh(bboxes_xyxy)
            return img, bboxes_cxcywh
    return img, bboxes_cxcywh

def draw_bboxes(img, bboxes_cxcywh):
    img_height, img_width = img.shape[:2]
    bboxes_xyxy = cxcywh2xyxy(bboxes_cxcywh)
    
    bboxes_xyxy[:, [0, 2]] *= img_width
    bboxes_xyxy[:, [1, 3]] *= img_height

    for bbox_xyxy in bboxes_xyxy:
        cv2.rectangle(img,
                      (int(bbox_xyxy[0]), int(bbox_xyxy[1])),
                      (int(bbox_xyxy[2]), int(bbox_xyxy[3])),
                      (0, 255, 0),2)
        