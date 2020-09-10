# siim-isic-melanoma-2020
Pytorch lightning code for SIIM-ISIC Melanoma Classification https://www.kaggle.com/c/siim-isic-melanoma-classification/data

### Requirements:
 - python==3.7.7 
 - pytorch==1.5, torchvision (cuda==10.2), pytorch-lightining==0.7.6
 - albumentations==0.4.5 (from sources)
 - opencv==3.4.2, matplotlib==3.1.3, tensorboard
 - pandas, jupyter, scikit-learn
 - g++, apex (from sources)
 - pip install efficientnet_pytorch 
 
 ### TODO:
  - aug-mix
  - AutoAugment (https://www.kaggle.com/nxrprime/siim-eda-augmentations-model-seresnet-unet/)
  - other features & targets (MLP)
  - - drop each feature is set to missing with this prob
  - Kfold (stratified: https://www.kaggle.com/zzy990106/pytorch-5-fold-efficientnet-baseline/notebook?select=auto_augment.py)
  - data echoing (https://arxiv.org/abs/1907.05550)
  - try hyper-opt/optuna (focal loss weight, batch, emb-size)
  - tta (multi_crop=5, flips, scale, )
  - ensembling
 
