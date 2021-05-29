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
    org_img_h, org_img_w = img.shape[:2]
    target_w, target_h = dsize

    ratio = min(target_w / org_img_w, target_h / org_img_h)
    
    non_padded_resized_h = round(org_img_h * ratio)
    non_padded_resized_w = round(org_img_w * ratio)

    img = cv2.resize(img, dsize=(non_padded_resized_w, non_padded_resized_h))

    pad_left = (target_w - non_padded_resized_w) // 2
    pad_right = target_w - non_padded_resized_w - pad_left
    pad_top = (target_h - non_padded_resized_h) // 2
    pad_bottom = target_h - non_padded_resized_h - pad_top

    # padding
    img = cv2.copyMakeBorder(img,
                             pad_top,
                             pad_bottom,
                             pad_left,
                             pad_right,
                             cv2.BORDER_CONSTANT,
                             value=(127, 127, 127))
    
    assert img.shape[0] == target_h and img.shape[1] == target_w
    
    if bboxes_cxcywh is not None:
        # padding으로 인한 객체 translation 보상
        bboxes_cxcywh[:, [0, 2]] *= non_padded_resized_w
        bboxes_cxcywh[:, [1, 3]] *= non_padded_resized_h

        bboxes_cxcywh[:, 0] += pad_left
        bboxes_cxcywh[:, 1] += pad_top

        bboxes_cxcywh[:, [0, 2]] /= target_w
        bboxes_cxcywh[:, [1, 3]] /= target_h

        return img, bboxes_cxcywh, [org_img_w, org_img_h], [pad_left, pad_top, pad_right, pad_bottom]
    return img

def horizontal_flip(img, bboxes_cxcywh, p=0.5):
    if random.random() < p:
        img = cv2.flip(img, 1)#1이 호리즌탈 방향 반전
        bboxes_cxcywh[:, 0] = 1. - bboxes_cxcywh[:, 0]
        return img, bboxes_cxcywh
    return img, bboxes_cxcywh

def random_translation(img, bboxes_cxcywh, p=1.0, border_value=(127, 127, 127)):
    if random.random() < p:
        img_h, img_w = img.shape[0:2]
        
        bboxes_xyxy = cxcywh2xyxy(bboxes_cxcywh)
        
        min_tx = round(img_w * np.min(bboxes_xyxy[:, 0]))
        max_tx = img_w-round(img_w * np.max(bboxes_xyxy[:, 2]))

        min_ty = round(img_h * np.min(bboxes_xyxy[:, 1]))
        max_ty = img_h-round(img_h * np.max(bboxes_xyxy[:, 3]))

        tx = random.randint(-min_tx, max_tx)
        ty = random.randint(-min_ty, max_ty)

        # translation matrix
        tm = np.float32([[1, 0, tx],
                         [0, 1, ty]])  # [1, 0, tx], [1, 0, ty]

        img = cv2.warpAffine(img, tm, (img_w, img_h), borderValue=border_value)

        bboxes_xyxy[:, [0, 2]] += (tx / img_w)
        bboxes_xyxy[:, [1, 3]] += (ty / img_h)
        bboxes_xyxy = np.clip(bboxes_xyxy, 0., 1.)

        bboxes_cxcywh = xyxy2cxcywh(bboxes_xyxy)
        return img, bboxes_cxcywh
    return img, bboxes_cxcywh
    
def random_scale(img, bboxes_cxcywh, p=1.0, border_value=(127, 127, 127), lower_bound=0.25, upper_bound=4.0, bbox_wmin_thr=32, bbox_hmin_thr=32):
    if random.random() < p:
        img_h, img_w = img.shape[0:2]
        
        bboxes_xyxy = cxcywh2xyxy(bboxes_cxcywh)
        
        # calculate range
        min_bbox_w = np.min(bboxes_cxcywh[:, 2]) * img_w
        min_bbox_h = np.min(bboxes_cxcywh[:, 3]) * img_h
        
        if min_bbox_w >= bbox_wmin_thr and min_bbox_h >= bbox_hmin_thr:
            min_scale = max(bbox_wmin_thr/min_bbox_w, bbox_hmin_thr/min_bbox_h)
        else:
            min_scale = 1.
            
        max_bbox_w = np.max(bboxes_cxcywh[:, 2]) * img_w
        max_bbox_h = np.max(bboxes_cxcywh[:, 3]) * img_h
        
        max_scale = max(img_w/max_bbox_w, img_h/max_bbox_h)
        
        # clip
        min_scale = max(min_scale, lower_bound)#lower bound
        max_scale = min(max_scale, upper_bound)#upper bound
        
        cx = img_w//2
        cy = img_h//2
        
        bboxes_xmin = round(img_w * np.min(bboxes_xyxy[:, 0]))
        bboxes_ymin = round(img_h * np.min(bboxes_xyxy[:, 1]))
        bboxes_xmax = round(img_w * np.max(bboxes_xyxy[:, 2]))
        bboxes_ymax = round(img_h * np.max(bboxes_xyxy[:, 3]))

        for _ in range(50):
            random_scale = random.uniform(min_scale, max_scale)
            
            #센터 기준으로 확대 혹은 축소
            tx = cx - random_scale * cx
            ty = cy - random_scale * cy
            
            transformed_bboxes_xmin = bboxes_xmin * random_scale + tx
            transformed_bboxes_ymin = bboxes_ymin * random_scale + ty
            transformed_bboxes_xmax = bboxes_xmax * random_scale + tx
            transformed_bboxes_ymax = bboxes_ymax * random_scale + ty
          
            if transformed_bboxes_xmin < 0 or transformed_bboxes_xmax >= img_w:
                continue

            if transformed_bboxes_ymin < 0 or transformed_bboxes_ymax >= img_h:
                continue
            
            # scale matrix
            sm = np.float32([[random_scale, 0, tx],
                            [0, random_scale, ty]])  # [1, 0, tx], [1, 0, ty]

            img = cv2.warpAffine(img, sm, (img_w, img_h), borderValue=border_value)

            bboxes_xyxy *= random_scale
            bboxes_xyxy[:, [0, 2]] += (tx / img_w)
            bboxes_xyxy[:, [1, 3]] += (ty / img_h)
            bboxes_xyxy = np.clip(bboxes_xyxy, 0., 1.)
            
            bboxes_cxcywh = xyxy2cxcywh(bboxes_xyxy)
            return img, bboxes_cxcywh
    return img, bboxes_cxcywh

def random_rotation(img, bboxes_cxcywh, angle=3., trial=40, p=1.0, border_value=(127, 127, 127)):
    if random.random() < p:
        img_h, img_w = img.shape[0:2]
        
        bboxes_xyxy = cxcywh2xyxy(bboxes_cxcywh) #tl br
        bboxes_xyxy[:, [0, 2]] *= img_w
        bboxes_xyxy[:, [1, 3]] *= img_h

        num_bboxes = len(bboxes_xyxy)
        bboxes_xyxyxyxy = np.ones((num_bboxes, 4, 3)) #tl tr bl br
        
        bboxes_xyxyxyxy[:, 0, [0, 1]] = bboxes_xyxy[:, [0, 1]]# tl
        bboxes_xyxyxyxy[:, 1, [0, 1]] = bboxes_xyxy[:, [2, 1]]# tr
        bboxes_xyxyxyxy[:, 2, [0, 1]] = bboxes_xyxy[:, [0, 3]]# bl
        bboxes_xyxyxyxy[:, 3, [0, 1]] = bboxes_xyxy[:, [2, 3]]# br
        
        bboxes_xyxyxyxy = bboxes_xyxyxyxy.reshape(-1, 3)
        
        for _ in range(trial):
            angle = random.uniform(-angle, angle)
            rm = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), angle=angle, scale=1.0) # rotation matrix

            rotated_bboxes_xyxyxyxy = (rm @ bboxes_xyxyxyxy.T).T
            rotated_bboxes_xyxyxyxy = rotated_bboxes_xyxyxyxy.reshape((num_bboxes, 4, 2))
            
            rotated_bboxes_xyxy = np.zeros_like(bboxes_xyxy)

            rotated_bboxes_xyxy[:, 0] = np.min(rotated_bboxes_xyxyxyxy[..., 0], axis=1)
            rotated_bboxes_xyxy[:, 1] = np.min(rotated_bboxes_xyxyxyxy[..., 1], axis=1)
            rotated_bboxes_xyxy[:, 2] = np.max(rotated_bboxes_xyxyxyxy[..., 0], axis=1)
            rotated_bboxes_xyxy[:, 3] = np.max(rotated_bboxes_xyxyxyxy[..., 1], axis=1)
            
            if np.min(rotated_bboxes_xyxy) < 0:
                continue
            
            if np.max(rotated_bboxes_xyxy[:, [0, 2]]) >= img_w or np.max(rotated_bboxes_xyxy[:, [1, 3]]) >= img_h:
                continue
            
            rotated_bboxes_xyxy[:, [0, 2]] /= img_w
            rotated_bboxes_xyxy[:, [1, 3]] /= img_h
            
            img = cv2.warpAffine(img, rm, (img_w, img_h), flags=cv2.INTER_LINEAR, borderValue=border_value)
            bboxes_cxcywh = xyxy2cxcywh(rotated_bboxes_xyxy)
            return img, bboxes_cxcywh
    return img, bboxes_cxcywh

def random_crop(img, bboxes_cxcywh, bboxes_class, trial=50, p=1.0, border_value=127):
    if random.random() < p:
        img_h, img_w = img.shape[0:2]
        
        bboxes_xyxy = cxcywh2xyxy(bboxes_cxcywh) #tl br
        bboxes_xyxy[:, [0, 2]] *= img_w
        bboxes_xyxy[:, [1, 3]] *= img_h

        bboxes_w = bboxes_cxcywh[:, 2] * img_w
        bboxes_h = bboxes_cxcywh[:, 3] * img_h
        bboxes_area = bboxes_w * bboxes_h
        
        for _ in range(trial):
            cropped_w = random.randint(img_w//8, img_w)
            cropped_h = random.randint(img_h//8, img_h)
            
            cropped_xmin = random.randint(0, max(img_w-cropped_w-1,1))
            cropped_ymin = random.randint(0, max(img_h-cropped_h-1,1))
            
            cropped_xmax = cropped_xmin + cropped_w
            cropped_ymax = cropped_ymin + cropped_h
            
            cropped_bboxes_xyxy = bboxes_xyxy.copy()
            cropped_bboxes_xyxy[:, [0, 2]] = np.clip(cropped_bboxes_xyxy[:, [0, 2]], a_min=cropped_xmin, a_max=cropped_xmax)
            cropped_bboxes_xyxy[:, [1, 3]] = np.clip(cropped_bboxes_xyxy[:, [1, 3]], a_min=cropped_ymin, a_max=cropped_ymax)
            
            cropped_bboxes_w = cropped_bboxes_xyxy[:, 2] - cropped_bboxes_xyxy[:, 0]
            cropped_bboxes_h = cropped_bboxes_xyxy[:, 3] - cropped_bboxes_xyxy[:, 1]
            cropped_bboxes_area = cropped_bboxes_w * cropped_bboxes_h
            
            iou = (cropped_bboxes_area/bboxes_area)
            size_constraint = (cropped_bboxes_area > 0) & (cropped_bboxes_w >= 3) & (cropped_bboxes_h >= 3) & (iou > 0.1)
            
            cropped_bboxes_xyxy = cropped_bboxes_xyxy[size_constraint]
            if len(cropped_bboxes_xyxy) == 0: continue

            iou = iou[size_constraint]
            if np.count_nonzero(iou < 0.9) > 0: continue

            mask = np.zeros_like(img)
            mask[cropped_ymin:cropped_ymax, cropped_xmin:cropped_xmax] = 1
            
            img[mask==0] = border_value
            cropped_bboxes_xyxy[:, [0, 2]] /= img_w
            cropped_bboxes_xyxy[:, [1, 3]] /= img_h
            
            bboxes_cxcywh = xyxy2cxcywh(cropped_bboxes_xyxy)
            bboxes_class = bboxes_class[size_constraint]
            
            return img, bboxes_cxcywh, bboxes_class
    return img, bboxes_cxcywh, bboxes_class

def mosaic(img, bboxes_cxcywh, bboxes_class, dataset, keep_ratio=True, bbox_wmin_thr=32, bbox_hmin_thr=32, trial=40, p=0.5):
    if random.random() <= p:
        img_h, img_w = img.shape[:2]
        resized_img_h, resized_img_w = img_h//2, img_w//2
        
        bbox_wmin = int(resized_img_w * np.min(bboxes_cxcywh[:, 2]))
        bbox_hmin = int(resized_img_h * np.min(bboxes_cxcywh[:, 3]))
        
        if bbox_wmin < bbox_wmin_thr or bbox_hmin < bbox_hmin_thr:
            return img, bboxes_cxcywh, bboxes_class
    
        mosaic_img = np.zeros_like(img)
        mosaic_bboxes_cxcywh = []
        mosaic_bboxes_class = []
        
        img = cv2.resize(img, dsize=(resized_img_w, resized_img_h))
        bboxes_cxcywh /= 2.
        
        mosaic_img[:resized_img_h, :resized_img_w] = img
        mosaic_bboxes_cxcywh.append(bboxes_cxcywh)
        mosaic_bboxes_class.append(bboxes_class)
        
        top_left_x = [resized_img_w, 0, resized_img_w]
        top_left_y = [0, resized_img_h, resized_img_h]
        tx = [0.5, 0, 0.5]
        ty = [0, 0.5, 0.5]
        
        i = 0
        for _ in range(trial):
            rand_idx = random.randint(0, len(dataset)-1)
            
            img, label = dataset[rand_idx]
            
            bboxes_class = label[:, 0].reshape(-1, 1)
            bboxes_cxcywh = label[:, 1:].reshape(-1, 4)
            
            if keep_ratio:
                img, bboxes_cxcywh, _, _ = aspect_ratio_preserved_resize(img, dsize=(resized_img_w, resized_img_h), bboxes_cxcywh=bboxes_cxcywh)
            else:
                img = cv2.resize(img, dsize=(resized_img_w, resized_img_h))
            
            bbox_wmin = int(resized_img_w * np.min(bboxes_cxcywh[:, 2]))
            bbox_hmin = int(resized_img_h * np.min(bboxes_cxcywh[:, 3]))
            
            if bbox_wmin < bbox_wmin_thr or bbox_hmin < bbox_hmin_thr:
                continue
            
            mosaic_img[top_left_y[i]:top_left_y[i]+resized_img_h, top_left_x[i]:top_left_x[i]+resized_img_w] = img
            bboxes_cxcywh /= 2.
  
            bboxes_cxcywh[:, 0] += tx[i]
            bboxes_cxcywh[:, 1] += ty[i]
                        
            mosaic_bboxes_cxcywh.append(bboxes_cxcywh)
            mosaic_bboxes_class.append(bboxes_class)

            if i == 2:
                break
            
            i += 1

        mosaic_bboxes_cxcywh = np.concatenate(mosaic_bboxes_cxcywh, axis=0)
        mosaic_bboxes_class = np.concatenate(mosaic_bboxes_class, axis=0)
        
        mosaic_bboxes_cxcywh = np.clip(mosaic_bboxes_cxcywh, 0., 1.)
        return mosaic_img, mosaic_bboxes_cxcywh, mosaic_bboxes_class
    return img, bboxes_cxcywh, bboxes_class

def mixup(img, bboxes_cxcywh, bboxes_class, dataset, keep_ratio=True, use_mosaic=False, alpha=8.0, beta=8.0, p=0.5, mosaic_p=0.5):
    if random.random() <= p:
        img_h, img_w = img.shape[:2]
            
        r = np.random.beta(alpha, beta)
        rand_idx = random.randint(0, len(dataset)-1)
        mixed_img, mixed_label = dataset[rand_idx]
            
        mixed_bboxes_class = mixed_label[:, 0].reshape(-1, 1)
        mixed_bboxes_cxcywh = mixed_label[:, 1:].reshape(-1, 4)
        
        if keep_ratio:
            mixed_img, mixed_bboxes_cxcywh, _, _ = aspect_ratio_preserved_resize(mixed_img, dsize=(img_w, img_h), bboxes_cxcywh=mixed_bboxes_cxcywh)
        else:
            mixed_img = cv2.resize(mixed_img, dsize=(img_w, img_h))
        
        if use_mosaic:
            mixed_img, mixed_bboxes_cxcywh, mixed_bboxes_class = mosaic(mixed_img, mixed_bboxes_cxcywh, mixed_bboxes_class, dataset, keep_ratio, p=mosaic_p)
        
        img = (img * r + mixed_img * (1 - r)).astype(np.uint8)
        bboxes_cxcywh = np.concatenate([bboxes_cxcywh, mixed_bboxes_cxcywh], axis=0)
        bboxes_class = np.concatenate([bboxes_class, mixed_bboxes_class], axis=0)
        return img, bboxes_cxcywh, bboxes_class
    return img, bboxes_cxcywh, bboxes_class
        
        
def augment_hsv(img, hgain=0.0138, sgain=0.664, vgain=0.464): # https://github.com/ultralytics/yolov5/blob/77415a42e5975ea356393c9f1d5cff0ae8acae2c/data/hyp.finetune.yaml
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

def cutout(img, max_n_holes=5):
    n_holes = random.randint(1, max_n_holes)
    
    img_h, img_w = img.shape[:2]
    
    erased_region_wmax = int(img_w * 0.2) 
    erased_region_hmax = int(img_h * 0.2) 
    
    mask = np.zeros((img_h, img_w),dtype=np.bool)
    for _ in range(n_holes):
        erased_region_w = random.randint(0, erased_region_wmax)
        erased_region_h = random.randint(0, erased_region_hmax)
        
        y = np.random.randint(img_h)
        x = np.random.randint(img_w)

        y1 = np.clip(y - erased_region_h // 2, 0, img_h)
        y2 = np.clip(y + erased_region_h // 2, 0, img_h)
        x1 = np.clip(x - erased_region_w // 2, 0, img_w)
        x2 = np.clip(x + erased_region_w // 2, 0, img_w)

        mask[y1:y2, x1:x2] = True

    img[mask] = 127
    return img

def draw_bboxes(img, bboxes_cxcywh):
    img_h, img_w = img.shape[:2]
    bboxes_xyxy = cxcywh2xyxy(bboxes_cxcywh)
    
    bboxes_xyxy[:, [0, 2]] *= img_w
    bboxes_xyxy[:, [1, 3]] *= img_h

    for bbox_xyxy in bboxes_xyxy:
        cv2.rectangle(img,
                      (int(bbox_xyxy[0]), int(bbox_xyxy[1])),
                      (int(bbox_xyxy[2]), int(bbox_xyxy[3])),
                      (0, 255, 0),2)
        