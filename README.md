# siim-isic-melanoma-2020
Pytorch lightning code for SIIM-ISIC Melanoma Classification https://www.kaggle.com/c/siim-isic-melanoma-classification/data

### Requirements:
 - python==3.7.7
 - pytorch==1.5, torchvision (cuda==10.2)
 - albumentations==0.1.8
 - opencv==3.4.2, matplotlib==3.1.3, tensorboard
 - pandas, jupyter, scikit-learn
 - apex (conda install -c conda-forge nvidia-apex)
 
 ### TODO:
  - FP-16
  - Albu ( CutOut)
  - aug-mix
  - loss weighting
  - AutoAugment
  - other features & targets (MLP)
  - - drop each feature is set to missing with this prob
  - Kfold
  - Efficient Net
  - focal loss from retinanet
  - data echoing (https://arxiv.org/abs/1907.05550)
  - prepare data (resize, numpy)
  - try hyper-opt
  - ensembling