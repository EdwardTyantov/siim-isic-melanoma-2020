from .resnet import resnext50_32x4d_ssl, resnext50_32x4d_fc
from .efficient_net import efficientnetb4_classic_fc

__all__ = ['factory', ]


def resnext50_32x4d_ssl_finetune(pretrained):
    net = resnext50_32x4d_ssl(pretrained=pretrained)
    layers = {
        'trained': [(net.layer4, 1),(net.layer3, 0.5) ],
        'untrained': [(net.fc, 1), ],
    }
    return net, layers


def resnext50_32x4d_ssl_fc(embedding=256, pretrained=True):
    net = resnext50_32x4d_fc(num_classes=1, embedding_len=embedding, pretrained=pretrained)
    layers = {
        'trained': [(net.layer4, 1), (net.layer3, 0.5)],
        'untrained': [(net.emb, 1), (net.fc, 1)],
    }
    return net, layers


def efficientnetb4_fc(embedding=256, pretrained=True):
    net = efficientnetb4_classic_fc(embedding_size=embedding, num_classes=1, pretrained=pretrained)
    layers = {
        # 'trained': [(net.layer4, 1), (net.layer3, 0.5)],
        'trained': [(layer, 1) for layer in net._blocks[-8:]] + [(layer, 0.5) for layer in net._blocks[-20:-8]],
        'untrained': [(net._emb, 1), (net._fc, 1)],
    }
    return net, layers


def factory(name, pretrained=True, **kwargs):
    """
    Returns model of the given network and layers to optimize, if layers is None it's further treated as model.parameters().
    layers = {'trained'/'untrained': [(layer params, layer_multiplier)]
    layer_lr = overall_lr * layer_multiplier
    if trained layer
    """
    print('Loading model', name)
    model_func = globals().get(name, None)
    if model_func is None:
        raise AttributeError("Model %s doesn't exist" % (name,))

    model, layers = model_func(pretrained=pretrained, **kwargs)
    setattr(model, 'name', name)

    return model, layers
