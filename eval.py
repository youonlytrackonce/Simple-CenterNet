from models import centernet
from data import dataset
from utils import common
from evaluation import metric

import numpy as np
import torch
import cv2

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CenterNet Detection')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Batch size for training')

    parser.add_argument('--img-w', default=512, type=int)
    parser.add_argument('--img-h', default=512, type=int)

    parser.add_argument('--weights', type=str, default="", help='load weights to resume training')
    parser.add_argument('--data', type=str, default="voc0712.yaml")
    parser.add_argument('--num-workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--flip', action='store_true')
    
    opt = parser.parse_args()
    common.mkdir(dir="gt", remove_existing_dir=True)
    common.mkdir(dir="pred", remove_existing_dir=True)
    
    dataset_dict = common.parse_yaml(opt.data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = centernet.CenterNet(num_classes=len(dataset_dict['classes']), pretrained_backbone=True)
    common.load_only_model_weights(model=model, weights_path=opt.weights, device=device)
        
    model.eval()
    model = model.to(device=device)
    
    test_set = dataset.DetectionDataset(root=dataset_dict['root'], 
                                        dataset_name=dataset_dict['dataset_name'], 
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

            batch_output = model(batch_img, flip=opt.flip)
            batch_output = model.post_processing(batch_output, batch_org_img_shape, batch_padded_ltrb, confidence_threshold=1e-2)
            
            for i in range(len(batch_img)):
                idx = batch_idx[i] # data index
                
                org_img_shape = batch_org_img_shape[i] # (w, h)
                padded_ltrb = batch_padded_ltrb[i]
                
                target_bboxes = batch_label[i]#.numpy()

                pred_bboxes = batch_output[i]
                target_bboxes = common.reconstruct_bboxes(normalized_bboxes=target_bboxes,
                                                        resized_img_shape=(model.img_w, model.img_h),
                                                        padded_ltrb=padded_ltrb,
                                                        org_img_shape=org_img_shape)
                target_bboxes = target_bboxes.numpy()

                gt_bboxes_batch.append(target_bboxes)

                img = cv2.imread(test_set.dataset.images_path[idx])
                
                img_file = os.path.basename(test_set.dataset.images_path[idx])
                txt_file = img_file.replace(".jpg", ".txt")
                
                gt_txt_file = os.path.join("gt", txt_file)
                pred_txt_file = os.path.join("pred", txt_file)
                
                common.write_bboxes(gt_txt_file, img, target_bboxes, dataset_dict['classes'], draw_rect=False)
  
                with open(pred_txt_file, "w") as f:
                    if pred_bboxes["num_detected_bboxes"] > 0:
                        pred_bboxes = np.concatenate([pred_bboxes["class"].reshape(-1, 1), 
                                                    pred_bboxes["position"].reshape(-1, 4),
                                                    pred_bboxes["confidence"].reshape(-1, 1)], axis=1)
            
                        class_tp_fp_score = metric.measure_tpfp(pred_bboxes, target_bboxes, 0.5, bbox_format='cxcywh')
                        class_tp_fp_score_batch.append(class_tp_fp_score)
                        
                        common.write_bboxes(pred_txt_file, img, pred_bboxes, dataset_dict['classes'], draw_rect=True)
                        
                    #cv2.imshow('img', img)
                    #cv2.waitKey(1)
                
        mean_ap, ap_per_class = metric.compute_map(class_tp_fp_score_batch, gt_bboxes_batch, num_classes=model.num_classes)
        for i in range(len(dataset_dict['classes'])):
            print("Class: ", dataset_dict['classes'][i], ", AP: ", np.round(ap_per_class[i], decimals=4))
        print("mAP: ", np.round(mean_ap, decimals=4))
        