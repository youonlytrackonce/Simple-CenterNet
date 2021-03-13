from models import centernet
from utils import tool
from utils import dataset

import torch
from torch.utils.tensorboard import SummaryWriter

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLO-v3 tiny Detection')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='Batch size for training')

    parser.add_argument('--img-w', default=512, type=int)
    parser.add_argument('--img-h', default=512, type=int)

    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--weights', type=str, default="", help='load weights to resume training')
    parser.add_argument('--total-epoch', type=int, default=150, help='total_epoch')
    parser.add_argument('--root', default="./dataset/VOCDevkit", help='Location of dataset directory')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='Weight decay for SGD')
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
    
    training_set = dataset.DetectionDataset(root="./dataset/VOCDevkit", dataset_name="voc", set="train")
    num_training_set_imgs = len(training_set)
    
    training_set_loader = torch.utils.data.DataLoader(training_set, 
                                                      opt.batch_size,
                                                      num_workers=opt.num_workers,
                                                      shuffle=True,
                                                      collate_fn=dataset.collate_fn,
                                                      pin_memory=True,
                                                      drop_last=True)
    
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=opt.lr, 
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay
                                )
    
    
    iterations_per_epoch = num_training_set_imgs // opt.batch_size 
    total_iteration = iterations_per_epoch * opt.total_epoch

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iteration)
    
    writer = SummaryWriter()
    
    start_epoch = 0
    n_iter = 0
    for epoch in range(start_epoch, opt.total_epoch):
        model.train()
        for i, batch_data in enumerate(training_set_loader):
            n_iter = (iterations_per_epoch * epoch) + i
            
            batch_img = batch_data["img"].to(device)
            batch_label = batch_data["label"]
            
            #forward
            batch_output = model(batch_img)
            loss, losses = model.compute_loss(batch_output, batch_label)
            
            writer.add_scalar('train/loss_offset_xy', losses[0].item(), n_iter)
            writer.add_scalar('train/loss_wh', losses[1].item(), n_iter)
            writer.add_scalar('train/loss_class_heatmap', losses[2].item(), n_iter)
            writer.add_scalar('train/loss', loss.item(), n_iter)
            writer.add_scalar('train/lr', tool.get_lr(optimizer), n_iter)

            #backword
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            scheduler.step()
        
        checkpoint = {
        'epoch': epoch,# zero indexing
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'mAP': 0.,
        'best_mAP': 0.
        }
        
        torch.save(checkpoint, os.path.join(opt.save_folder, 'epoch_' + str(epoch + 1) + '.pth'))
        