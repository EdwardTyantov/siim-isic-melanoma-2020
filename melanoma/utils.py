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
        
    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in ('optimizer', '_callback')}

