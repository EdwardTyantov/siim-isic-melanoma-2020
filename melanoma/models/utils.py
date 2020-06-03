import sys
from .resnet import resnext50_32x4d_ssl

__all__ = ['factory', ]


def resnext50_32x4d_ssl_finetune(pretrained):
    net = resnext50_32x4d_ssl(pretrained=pretrained)
    layers = {
        'trained': [ (net.layer4, 1), ],
        'untrained': [(net.fc, 1), ],
    }
    return net, layers
    

def factory(name, pretrained=True, **kwargs):
    """
    Returns model of the given network and layers to optimize, if layers is None it's further treated as model.parameters().
    layers = {'trained'/'untrained': [(layer params, layer_multiplier)]
    layer_lr = overall_lr * layer_multiplier
    if trained layer
    """
    model_func = globals().get(name, None)
    if model_func is None:
        raise AttributeError("Model %s doesn't exist" % (name,))

    model, layers = model_func(pretrained=pretrained, **kwargs)
    setattr(model, 'name', name)

    return model, layers
