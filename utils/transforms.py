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

def random_translation(img, bboxes_cxcywh, p=0.5, border_value=(127, 127, 127)):
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
    
    
    
    
    
