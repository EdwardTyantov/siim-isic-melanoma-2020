import sys
import torch, torch.nn as nn
from torch.utils import model_zoo
from efficientnet_pytorch import EfficientNet, get_model_params
from efficientnet_pytorch.utils import url_map

__all__ = ['efficientnetb4_classic_fc', ]


class EfficientNetFC(EfficientNet):
    def __init__(self, embedding_size, blocks_args=None, global_params=None):
        super().__init__(blocks_args, global_params)
        self._emb = nn.Sequential(nn.Linear(self._fc.in_features, embedding_size),
                                 nn.ReLU(inplace=True),
                                 #nn.Dropout(p=0.2),
                                 nn.BatchNorm1d(embedding_size)
                                 )
        self._fc = nn.Linear(embedding_size, global_params.num_classes)
        
    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        x = self.extract_features(inputs)
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._emb(x)
        x = self._fc(x)
        return x
    
    @classmethod
    def from_name(cls, model_name, embedding_size, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(embedding_size, blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, embedding_size, advprop=False, num_classes=1000):
        model = cls.from_name(model_name, embedding_size, override_params={'num_classes': num_classes})
        state_dict = model_zoo.load_url(url_map[model_name])
        state_dict.pop('_fc.weight'); state_dict.pop('_fc.bias')
        model.load_state_dict(state_dict, strict=False)
        
        return model


def efficientnetb4_classic_fc(embedding_size=256, num_classes=1, pretrained=True):
    if pretrained:
        model = EfficientNetFC.from_pretrained('efficientnet-b4', embedding_size, num_classes=num_classes)
    else:
        raise NotImplementedError
    
    return model


def test():
    model = efficientnetb4_classic_fc()
    #model = EfficientNet.from_pretrained('efficientnet-b4')
    model.eval()
    #print(model)
    print(model._blocks[4:6])
    sys.exit()
    x = torch.randn((4, 3, 224, 224))
    r = model(x)
    print(r.size(), r)


if __name__ == '__main__':
    sys.exit(test())
