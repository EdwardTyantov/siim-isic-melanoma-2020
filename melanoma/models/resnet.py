import sys
import torch, torch.nn as nn
from torchvision.models.resnet import resnext50_32x4d, ResNet, Bottleneck


def resnext50_32x4d_ssl(num_classes=1, pretrained=True):
    
    if pretrained:
        resnet = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    else:
        resnet = resnext50_32x4d(pretrained=False, num_classes=1)
    
    return resnet


class ResNetFC(ResNet):
    def __init__(self, embedding_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb = nn.Sequential(nn.Linear(self.fc.in_features, embedding_len),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(embedding_len)
                                )
        self.fc = nn.Linear(embedding_len, kwargs['num_classes'])
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
    
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
    
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.emb(x)
        x = self.fc(x)
        return x


def resnext50_32x4d_fc(num_classes=1, embedding_len=256, pretrained=True):
    model = ResNetFC(embedding_len, Bottleneck, [3, 4, 6, 3], num_classes=num_classes, groups=32, width_per_group=4,
                     ) #zero_init_residual=True)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth')
        state_dict.pop('fc.weight'); state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    
    return model


class ResNetMultiHead(ResNetFC):
    def __init__(self, embedding_len, diagnosis_num, *args, **kwargs):
        super().__init__(embedding_len, *args, **kwargs)
        self.emb2 = nn.Sequential(nn.Linear(self.emb[0].in_features, embedding_len),
                                 nn.ReLU(inplace=True),
                                 nn.BatchNorm1d(embedding_len)
                                 )
        self.diagnosis_fc = nn.Linear(embedding_len, diagnosis_num)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        target = self.fc(self.emb(x))
        diagnosis = self.diagnosis_fc(self.emb2(x))
        
        return target, diagnosis


def resnext50_32x4d_2head(num_classes=1, diagnosis_num=6, embedding_len=256, pretrained=True):
    model = ResNetMultiHead(embedding_len, diagnosis_num, Bottleneck, [3, 4, 6, 3], num_classes=num_classes, groups=32,
                            width_per_group=4)
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            'https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_supervised_resnext50_32x4-ddb3e555.pth')
        state_dict.pop('fc.weight');
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    
    return model
    
def test():
    model = resnext50_32x4d_fc()
    # construct an optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.Adam(
        [
            {"params": model.layer4.parameters(), "lr": 1e-3},
        ],
        lr=5e-4,
    )
    print(model)
    model.eval()
    x = torch.randn((4, 3, 224, 224))
    r = model(x)
    print(r.size(), r)


if __name__ == '__main__':
    sys.exit(test())
