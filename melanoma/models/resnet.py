import sys
import torch, torch.nn as nn
from torchvision.models.resnet import resnext50_32x4d

__all__ = ['resnext50_32x4d_ssl', ]


def resnext50_32x4d_ssl(num_classes=1, pretrained=True):
    
    if pretrained:
        resnet = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    else:
        resnet = resnext50_32x4d(pretrained=False, num_classes=1)
    
    return resnet
    
    
def test():
    model = resnext50_32x4d_ssl()

    # сделайте так, чтобы мы обучали только нужные части сети
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    # construct an optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": 1e-3},
            {"params": model.fc.parameters(), "lr": 4e-2},
        ],
        lr=5e-4,
    )
    print(model)


if __name__ == '__main__':
    sys.exit(test())

