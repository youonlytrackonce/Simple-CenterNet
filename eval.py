from models import centernet
from utils import dataset
from utils import tool
from evaluation import metric

import numpy as np
import torch
import cv2

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO-v3 tiny Detection')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='Batch size for training')

    parser.add_argument('--img-w', default=512, type=int)
    parser.add_argument('--img-h', default=512, type=int)

    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--weights', type=str, default="", help='load weights to resume training')
    parser.add_argument('--root', default="./dataset/VOCDevkit", help='Location of dataset directory')
    parser.add_argument('--num-workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--save-folder', default='./weights', type=str, help='where you save weights')
    parser.add_argument('--seed', default=7777, type=int)

    opt = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = centernet.CenterNet(pretrained_backbone=True)
    if opt.weights is not None:
        chkpt = torch.load(opt.weights, map_location=device)
        model.load_state_dict(chkpt['model_state_dict'], strict=False)
    model.eval()
    model = model.to(device=device)
    
    test_set = dataset.DetectionDataset(root="C:/dataset/VOCDevkit", 
                                        dataset_name="voc", 
                                        set="test",
                                        img_w=opt.img_w, 
                                        img_h=opt.img_h,
                                        keep_ratio=True)
    
    test_set_loader = torch.utils.data.DataLoader(test_set, 
                                                  opt.batch_size,
                                                  num_workers=opt.num_workers,
                                                  shuffle=False,
                                                  collate_fn=dataset.collate_fn,
                                                  pin_memory=True,
                                                  drop_last=False)
    
    gt_bboxes_batch = []
    class_tp_fp_score_batch = []
    with torch.no_grad():

        for batch_data in test_set_loader:
            batch_img = batch_data["img"].to(device)
            batch_label = batch_data["label"]
            batch_idx = batch_data["idx"]
            batch_org_img_shape = batch_data["org_img_shape"]
            batch_padded_ltrb = batch_data["padded_ltrb"]

            batch_output = model(batch_img)
            batch_output = model.post_processing(batch_output, batch_org_img_shape, batch_padded_ltrb, confidence_threshold=1e-2)
            
            for i in range(len(batch_img)):
                idx = batch_idx[i] # data index
                
                org_img_shape = batch_org_img_shape[i] # (w, h)
                padded_ltrb = batch_padded_ltrb[i]
                
                target_bboxes = batch_label[i]#.numpy()

                pred_bboxes = batch_output[i]
                target_bboxes = tool.reconstruct_bboxes(normalized_bboxes=target_bboxes,
                                                        resized_img_shape=(model.img_w, model.img_h),
                                                        padded_ltrb=padded_ltrb,
                                                        org_img_shape=org_img_shape)
                target_bboxes = target_bboxes.numpy()

                # print(target_bboxes)
                gt_bboxes_batch.append(target_bboxes)

                img = cv2.imread(test_set.dataset.images_path[idx])
                

                if pred_bboxes["num_detected_bboxes"] > 0:
                    pred_bboxes = np.concatenate([pred_bboxes["class"].reshape(-1, 1), 
                                                pred_bboxes["position"].reshape(-1, 4),
                                                pred_bboxes["confidence"].reshape(-1, 1)], axis=1)

                    class_tp_fp_score = metric.measure_tpfp(pred_bboxes, target_bboxes, 0.5, bbox_format='cxcywh')
                    class_tp_fp_score_batch.append(class_tp_fp_score)
                    # print(np.min(pred_bboxes[:, 3]), np.min(pred_bboxes[:, 4]))
                    for pred_bbox in pred_bboxes:
                        # if pred_bbox[-1] < .1:
                        #     continue
                        
                        # print(pred_bbox[-1])
                        
                        #pred_bbox = pred_bbox.astype(np.int32)
                        
                        l = int(pred_bbox[1] - pred_bbox[3] / 2.)
                        r = int(pred_bbox[1] + pred_bbox[3] / 2.)

                        t = int(pred_bbox[2] - pred_bbox[4] / 2.)
                        b = int(pred_bbox[2] + pred_bbox[4] / 2.)

                        cv2.rectangle(img=img, pt1=(l, t), pt2=(r, b), color=(0, 255, 0), thickness=3)

                cv2.imshow('img', img)
                cv2.waitKey(1)
                
        mean_ap = metric.compute_map(class_tp_fp_score_batch, gt_bboxes_batch, num_classes=model.num_classes)
        print(mean_ap)