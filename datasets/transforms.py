import os
import cv2
import numpy as np
import torch
import torchvision
import copy

def mark_heatmap(gt_heatmap, x, y, w, h, class_id):
    """ (x, y)위치를 중심으로 w x h 크기의 가우시안 커널 값을 gt_heatmap 상에 붙여넣습니다.
    Args:
        gt_heatmap: ground truth heatmap, shape: (w, h, 1)
        x, y: 물체의 center x, y 
        w, h: 물체의 w, h
        class_id: 물체의 class id
        CenterNet Paper 3. Preliminary 참조
    """

    gt_heatmap_h, gt_heatmap_w, _ = gt_heatmap.shape

    l = np.maximum(x - w // 2, 0)
    r = np.minimum(x + w // 2, gt_heatmap_w)

    t = np.maximum(y - h // 2, 0)
    b = np.minimum(y + h // 2, gt_heatmap_h)

    #cropped area that will be marked
    cropped_gt_heatmap = gt_heatmap[t:b, l:r, class_id]
    cropped_gt_heatmap_h, cropped_gt_heatmap_w = cropped_gt_heatmap.shape

    sigma_x = cropped_gt_heatmap_w / 6.
    sigma_y = cropped_gt_heatmap_h / 6.

    grid_x = np.arange(cropped_gt_heatmap_w) - cropped_gt_heatmap_w // 2
    grid_y = np.arange(cropped_gt_heatmap_h) - cropped_gt_heatmap_h // 2

    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    #refer to centernet paper
    gaussian_1d_kernel_x = np.exp(-(grid_x * grid_x) / (2 * sigma_x * sigma_x))
    gaussian_1d_kernel_y = np.exp(-(grid_y * grid_y) / (2 * sigma_y * sigma_y))
    gaussian_2d_kernel = gaussian_1d_kernel_x * gaussian_1d_kernel_y

    cropped_gt_heatmap[:, :] = np.maximum(cropped_gt_heatmap[:, :], gaussian_2d_kernel)


def build_gt_tensors(img_size, 
                     gt_size, 
                     bboxes_regression_method,
                     num_classes, 
                     class_ids, 
                     bboxes):
    """ Loss 계산을 위하여 Tensor 형태의 Ground Truth 데이터 반환
    Args:
        img_size: image resolution (w, h)
        gt_size: ground truth heatmap resolution (gt_w=gt_size[0], gt_h=gt_size[1], num_classes), it depends on stride size.
        bboxes_regression_method: "ltrb" or "wh"
        num_classes: number of class
        class_ids: ground truth class, shape: (N)
        bboxes: ground truth heatmap, shape: (N, 4)
    Returns:
        gt_heatmap: ground truth heatmap (num_classes, gt_w, gt_h)
        gt_positive_mask: ground truth heatmap (gt_w, gt_h)
        gt_offset: ground truth offset (gt_w, gt_h, 2)
        gt_bboxes: ground truth bounding box width and height or ltrb distance from center point (gt_w, gt_h, 2) or (gt_w, gt_h, 4)
    """

    gt_heatmap = np.zeros(
        (gt_size[1], gt_size[0], num_classes),
        dtype=np.float32)  #CenterNet 논문 Y_{xyc}에 해당함, heatmap에 대한 GT
    gt_positive_mask = np.zeros(
        (gt_size[1], gt_size[0]),
        dtype=np.float32)  #CenterNet 논문 tilde{p}에 해당하는 위치를 1로 마스킹한 마스크
    gt_offset = np.zeros(
        (gt_size[1], gt_size[0], 2),
        dtype=np.float32)  #CenterNet 논문 eq(2)참조, Center 좌표의 오차 값을 보상하기 위한 GT
    
    # ltrb
    if bboxes_regression_method == "wh":
        gt_bboxes = np.zeros(
            (gt_size[1], gt_size[0], 2), dtype=np.float32
        )  #CenterNet 논문 eq(3)참조, 바운딩 박스의 Size를 Regression하기 위한 GT
    elif bboxes_regression_method == "ltrb":
        gt_bboxes = np.zeros((gt_size[1], gt_size[0], 4), dtype=np.float32)

    
    for class_id, bbox in zip(class_ids, bboxes):
        
        x, y, w, h = copy.deepcopy(bbox)#[center_x, center_y, w, h], range: [0, 1]
        
        x_gt_f32 = np.float32(x * gt_size[0])
        y_gt_f32 = np.float32(y * gt_size[1])
        w_gt_f32 = np.float32(w * gt_size[0])
        h_gt_f32 = np.float32(h * gt_size[1])

        x_gt = np.clip(round(x_gt_f32), 0, gt_size[0] - 1)
        y_gt = np.clip(round(y_gt_f32), 0, gt_size[1] - 1)
        w_gt = round(w_gt_f32)
        h_gt = round(h_gt_f32)
        
        mark_heatmap(gt_heatmap, x_gt, y_gt, w_gt, h_gt, class_id)

        gt_positive_mask[y_gt, x_gt] = 1.

        gt_offset[y_gt, x_gt, 0] = x_gt_f32 - x_gt
        gt_offset[y_gt, x_gt, 1] = y_gt_f32 - y_gt

        if bboxes_regression_method == "wh":
            gt_bboxes[y_gt, x_gt, 0] = w_gt_f32
            gt_bboxes[y_gt, x_gt, 1] = h_gt_f32
        elif bboxes_regression_method == "ltrb":
            xmin = x_gt_f32 - w_gt_f32 / 2.
            ymin = y_gt_f32 - h_gt_f32 / 2.
            xmax = x_gt_f32 + w_gt_f32 / 2.
            ymax = y_gt_f32 + h_gt_f32 / 2.

            gt_bboxes[y_gt, x_gt, 0] = x_gt_f32 - xmin  #l
            gt_bboxes[y_gt, x_gt, 1] = y_gt_f32 - ymin  #t
            gt_bboxes[y_gt, x_gt, 2] = xmax - x_gt_f32  #r
            gt_bboxes[y_gt, x_gt, 3] = ymax - y_gt_f32  #b

    # gt_heatmap shape을 convolutional layer의 output shape과 일치 시켜줌 (, H, W, 1) -> (, 1, H, W)
    #gt_heatmap = gt_heatmap.transpose(2, 0, 1)  # (1, gt_h, gt_w)
    #gt_heatmap = np.ascontiguousarray(gt_heatmap)

    return gt_heatmap, gt_positive_mask, gt_offset, gt_bboxes


def img2tensor(img):
    #the format of img needs to be bgr format

    img = img.astype(np.float32)
    img = img[..., ::-1]  #bgr2rgb
    img = img.transpose(2, 0, 1)  #(H, W, CH) -> (CH, H, W)
    img = np.ascontiguousarray(img)

    tensor = torch.tensor(img, dtype=torch.float32)
    return tensor

def letter_box_resize(img, dsize, class_ids=None, bboxes=None):
    
    original_height, original_width = img.shape[:2]
    target_width, target_height = dsize

    ratio = min(
        float(target_width) / original_width,
        float(target_height) / original_height)
    resized_height, resized_width = [
        round(original_height * ratio),
        round(original_width * ratio)
    ]

    img = cv2.resize(img, dsize=(resized_width, resized_height))

    pad_left = (target_width - resized_width) // 2
    pad_right = target_width - resized_width - pad_left
    pad_top = (target_height - resized_height) // 2
    pad_bottom = target_height - resized_height - pad_top

    # padding
    img = cv2.copyMakeBorder(img,
                             pad_top,
                             pad_bottom,
                             pad_left,
                             pad_right,
                             cv2.BORDER_CONSTANT,
                             value=(0, 0, 0))

    try:
        if img.shape[0] != target_height and img.shape[1] != target_width:  # 둘 중 하나는 같아야 함
            raise Exception('Letter box resizing method has problem.')
    except Exception as e:
        print('Exception: ', e)
        exit(1)

    if class_ids is not None and bboxes is not None:
        # padding으로 인한 객체 translation 보상
        bboxes[:, [0, 2]] *= resized_width
        bboxes[:, [1, 3]] *= resized_height

        bboxes[:, 0] += pad_left
        bboxes[:, 1] += pad_top

        bboxes[:, [0, 2]] /= target_width
        bboxes[:, [1, 3]] /= target_height

        return img, class_ids, bboxes
    return img
