# Simple-CenterNet

PyTorch Implementation of CenterNet([Object as Points](https://arxiv.org/abs/1904.07850))
- You don't need to bulid some cpp code to use Deformable Convolution used in CenterNet.

## Performance

## On VOC(Training:0712 trainval, Test:07)

|Repo| Backbone     | 0.5 mAP    | Trained model    |  
|:------------:|:-------:|:-------:|:-----------------:|  
|**This Repo**|ResNet-18|  78.1      |   |  
|[xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)|ResNet-18|75.6      |   |  
|[Ximilar-com/xcenternet](https://github.com/Ximilar-com/xcenternet)|ResNet-18     |  70.0      | |
|[xuannianz/keras-CenterNet](https://github.com/xuannianz/keras-CenterNet)|**ResNet-50**    |  72.9      | |
|[bubbliiiing/centernet-keras](https://github.com/bubbliiiing/centernet-keras)|**ResNet-50**     | 77.1      | |

## On COCO

|Repo| Backbone     |  mAP    | Trained model    |  
|:------------:|:-------:|:-------:|:-----------------:|  
|**This Repo**|ResNet-18|       |   |  
|[xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)|ResNet-18| 28.1      |   |  


## What's difference between paper and this repo?

### VOC

#### Paper
Refer to `Appendix D: Experiments on PascalVOC` in the paper
- Learning rate scheduler: MultiStepLR(milestones=[45, 60], gamma=0.1)
- Augmentation: HorizontalFlip, RandomScale, RandomTranslation, RandomCrop, and Color Jittering
- Kernel size of max pooling: 3

#### This Repo

- Learning rate scheduler: CosineDecay(per iteration)
- Augmentation: HorizontalFlip, RandomScale, RandomTranslation, RandomCrop, **Mosaic**, **Mixup(with Mosaic + 1.0 AP)**, and Color Jittering
- Kernel size of max pooling: 7
- Gaussian Kernel Generation Method: followed the method proposed in [Training-Time-Friendly Network for Real-Time Object Detection
](https://arxiv.org/abs/1909.00700)(It’s not carefully selected. I just think that it is more reasonable than original one.)

### COCO17

#### Paper

#### This Repo

## Setup
```
git clone https://github.com/developer0hye/Simple-CenterNet
cd Simple-CenterNet
```

if (your_os == 'Window'):
```
scripts/download-voc0712.bat
scripts/download-coco17.bat
```
else:
```
scripts/download-voc0712.sh
scripts/download-coco17.sh
```

## Training

### VOC07+12

```
python train.py --data ./data/voc0712.yaml --step-batch-size 32 --forward-batch-size 32 --total-epoch 70
```

If your gpu memory is too lower to train the model, you should try to reduce forward-batch-size.


### COCO17

## Evaluation

### VOC07+12
```
python eval.py --data ./data/voc0712.yaml --weights your_model.pth --flip
```

### COCO17

## Things we tried that didn't work

- Random Rotation Augmentation