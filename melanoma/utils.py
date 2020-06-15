import os, random, logging
import numpy as np
import torch

logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)


def init_seed(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, *args, callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback = callback
        
    def _reduce_lr(self, epoch):
        logger.info(f'Reducing lrs, epoch={epoch}')
        super()._reduce_lr(epoch)
        self._callback()
        
    def step(self, metrics, epoch=None):
        #print('LR:', float(metrics), epoch)
        super().step(metrics, epoch)
        
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_callback')}


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res
