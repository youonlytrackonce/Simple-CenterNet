from models import centernet
from utils import tool
from utils import dataset

import torch
from torch.utils.tensorboard import SummaryWriter

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO-v3 tiny Detection')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='Batch size for training')

    parser.add_argument('--img-w', default=512, type=int)
    parser.add_argument('--img-h', default=512, type=int)

    parser.add_argument('--lr', default=1.25e-4, type=float, help='initial learning rate')
    parser.add_argument('--weights', type=str, default="", help='load weights to resume training')
    parser.add_argument('--total-epoch', type=int, default=150, help='total_epoch')
    parser.add_argument('--root', default="./dataset/VOCDevkit", help='Location of dataset directory')
    parser.add_argument('--num-workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--save-folder', default='./weights', type=str, help='where you save weights')
    parser.add_argument('--seed', default=7777, type=int)

    opt = parser.parse_args()
    tool.setup_seed(opt.seed)
    
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = centernet.CenterNet(pretrained_backbone=True)
    model = model.to(device=device)
    
    training_set = dataset.DetectionDataset(root="C:/dataset/VOCDevkit",
                                            dataset_name="voc", 
                                            set="train",
                                            img_w=opt.img_w, img_h=opt.img_h,
                                            use_augmentation=True,
                                            keep_ratio=True)
    
    num_training_set_imgs = len(training_set)
    
    training_set_loader = torch.utils.data.DataLoader(training_set, 
                                                      opt.batch_size,
                                                      num_workers=opt.num_workers,
                                                      shuffle=True,
                                                      collate_fn=dataset.collate_fn,
                                                      pin_memory=True,
                                                      drop_last=True)
    
    initial_lr = 5e-4 * (opt.batch_size/128.)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.
    
    #     from models import dcn

    # params = []
    # deformable_conv_params = []
    
    # for m in model.children():
    #     if isinstance(m, dcn.DeformableConv2d):
    #         for p in m.offset_conv.parameters():
    #             deformable_conv_params.append(p)
                
    #         for p in m.modulator_conv.parameters():
    #             deformable_conv_params.append(p)
                
    #         for p in m.regular_conv.parameters():
    #             params.append(p)
    #     else:
    #         for p in m.parameters():
    #             params.append(p)

    # optimizer = torch.optim.Adam([{"params": params, "lr": opt.lr},
    #                               {"params": deformable_conv_params, "lr": opt.lr * 0.1},
    #                               ])
    
    
    iterations_per_epoch = num_training_set_imgs // opt.batch_size 
    total_iteration = iterations_per_epoch * opt.total_epoch

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iteration)
    warmup_iteration = 1000
    
    writer = SummaryWriter()
    
    start_epoch = 0
    if len(opt.weights) == 0:
            pass
    else:
        checkpoint = torch.load(opt.weights)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    for epoch in range(start_epoch, opt.total_epoch):
        model.train()
        for i, batch_data in enumerate(training_set_loader):
            n_iteration = (iterations_per_epoch * epoch) + i
            
            batch_img = batch_data["img"].to(device)
            batch_label = batch_data["label"]
            
            #forward
            batch_output = model(batch_img)
            loss, losses = model.compute_loss(batch_output, batch_label)
            
            writer.add_scalar('train/loss_offset_xy', losses[0].item(), n_iteration)
            writer.add_scalar('train/loss_wh', losses[1].item(), n_iteration)
            writer.add_scalar('train/loss_class_heatmap', losses[2].item(), n_iteration)
            writer.add_scalar('train/loss', loss.item(), n_iteration)
            writer.add_scalar('train/lr', tool.get_lr(optimizer), n_iteration)

            #backword
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if n_iteration > warmup_iteration:
                scheduler.step()
            else:
                lr = initial_lr * float(n_iteration) / warmup_iteration
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        
        checkpoint = {
        'epoch': epoch,# zero indexing
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'mAP': 0.,
        'best_mAP': 0.
        }
        
        torch.save(checkpoint, os.path.join(opt.save_folder, 'epoch_' + str(epoch + 1) + '.pth'))
        
