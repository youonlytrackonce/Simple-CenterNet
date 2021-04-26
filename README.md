# Simple-CenterNet

I re-implemented CenterNet([Object as Points](https://arxiv.org/abs/1904.07850)) using PyTorch. You don't need to bulid some cpp code to use Deformable Convolution used in CenterNet.

## Performance

|Repo| Backbone     |  Dataset    |  mAP    | trained model    |  
|:------------:|:------------:|:-------:|:-------:|:-----------------:|  
|Ours|ResNet-18| VOC(Training:07+12, Test: 07)    | 76.1      |   |  
|[xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)|ResNet-18| VOC(Training:07+12, Test: 07)    | 75.6      |   |  
|[Ximilar-com/xcenternet](https://github.com/Ximilar-com/xcenternet)|ResNet-18     | VOC(Training:07+12, Test: 07)    | 70.0      | |
|[xuannianz/keras-CenterNet](https://github.com/xuannianz/keras-CenterNet)|**ResNet-50**    | VOC(Training:07+12, Test: 07)    | 72.9      | |
|[bubbliiiing/centernet-keras](https://github.com/bubbliiiing/centernet-keras)|**ResNet-50**     | VOC(Training:07+12, Test: 07)    | 77.1      | |

## What's difference between paper and this repo?

### Paper
Refer tp `Appendix D: Experiments on PascalVOC` in the paper
- Epochs: 70:
- Learning rate scheduler: MultiStepLR(milestones=[45, 60], gamma=0.1)
- Augmentation: RandomScale, RandomTranslation, RandomCrop, Color Jittering
- Kernel size of max pooling: 3

### This Repo

- Epochs: 140
- Learning rate scheduler: CosineDecay(per iteration)
- Augmentation: RandomScale, RandomTranslation, Mosaic, Color Jittering, CutOut
- Kernel size of max pooling: 5
- Gaussian Kernel Generation Method: follows the method proposed in [Training-Time-Friendly Network for Real-Time Object Detection
](https://arxiv.org/abs/1909.00700)(It’s not carefully selected. I just think that it is more reasonable than original one.)

## Setup
```
git clone https://github.com/developer0hye/Simple-CenterNet
cd Simple-CenterNet
```

if (your_os == 'Window'):
```
scripts/download-voc0712.bat
```
else:
```
scripts/download-voc0712.sh
```

## Training(on VOC07+12)
```
python train.py --batch-size 32
```

## Evaluation(on VOC07)
```
python eval.py --batch-size 32 --weights your_model.pth
```
