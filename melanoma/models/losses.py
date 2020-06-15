import torch, torch.nn as nn
import torch.nn.functional as F

CE_MAPPING = {
    'bce': F.binary_cross_entropy_with_logits,
    'ce': F.cross_entropy,
}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, ce_func='bce', reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_func = CE_MAPPING[ce_func]
        self.reduce = reduce

    def forward(self, inputs, targets, weight=None):
        ce = self.ce_func(inputs, targets, weight=weight, reduction='none')
        pt = torch.exp(-ce)
        focal_weight = self.alpha * torch.pow(1 - pt, self.gamma) # TODO: ( 1 - alfa + raw prob and not exp for p_t)
        # https://github.com/BorisLestsov/MADE/blob/master/seminar7-detection2/retinanet/losses.py
        focal_loss = focal_weight * ce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
            
        return focal_loss
