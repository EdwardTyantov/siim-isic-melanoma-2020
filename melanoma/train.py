import os, sys, time, logging, argparse
import torch.distributed as dist
import numpy as np, pandas as pd
from hashlib import md5
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from config import IMAGE_DIR, TRAIN_CSV, TEST_CSV, RESULT_DIR
from utils import init_seed
from models import factory as model_factory
from models.losses import FocalLoss
from data.utils import Datasets
from utils import ReduceLROnPlateau


logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)


class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model, self.layers = model_factory(kwargs['model_name'])
        self.init_loss()
        for k,v in kwargs.items(): # maybe iterate them, but too much repetition
            setattr(self, k, v)
        self.hparams = kwargs
        self.is_ddp = self.distributed_backend in ('ddp', 'ddp2')
        self.num_workers = self.is_ddp and self.num_workers // self.gpus or self.num_workers
        
    def prepare_data(self) -> None:
        ds = Datasets(IMAGE_DIR, TRAIN_CSV, TEST_CSV, self.transform_name, self.image_size, p=self.p, val_split=0.2)
        self.train_dataset = ds.train_dataset
        self.val_dataset = ds.val_dataset
        self.test_dataset = ds.test_dataset
        
    def forward(self, batch):
        # TODO: add other features
        out = self.model(batch['image'])
        return out
    
    def init_loss(self):
        self.loss = FocalLoss(logits=True)

    def train_dataloader(self):
        # TODO: add sampler, great imbalance of zeros, don't forget to replace_sampler_ddp ?
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size*self.gpus, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size*self.gpus, shuffle=False,
                          num_workers=self.num_workers)
    
    def configure_optimizers(self):
        self.warmup = 0
        if self.layers:
            parameters = []
            for param in self.model.parameters():
                param.requires_grad = False
            for layer, scale in self.layers['untrained']:
                parameters.append({'params': layer.parameters(), 'lr': scale*self.lr})
                for param in layer.parameters():
                    param.requires_grad = True
            for layer, scale in self.layers['trained']:
                self.warmup = 1
                lr = scale * self.lr
                parameters.append({'params': layer.parameters(), 'lr': lr, 'after_warmup_lr': lr})
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info(str(parameters))
        else:
            parameters = self.model.parameters()

        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(parameters, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(parameters, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, patience=1, verbose=True, callback=self.load_best_model)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        return [optimizer], [scheduler]

    def load_best_model(self):
        "Don't try at home. This is done for ReduceLROnPlateau to load best weights, when reduce LR"
        checkpoint = self.trainer.checkpoint_callback
        if checkpoint is None:
            raise (Exception, 'Callback wasn\'t specified')
        tmp_file = '/tmp/djhasjkdaskjd.tmp'
        # the checkpoint contains best k models, let's extract best path
        if not (self.is_ddp and dist.get_rank() != 0):
            _op = min if checkpoint.mode == 'min' else max
            best_path = _op(checkpoint.best_k_models, key=checkpoint.best_k_models.get)
            with open(tmp_file, 'w') as wf: # TODO: refactor string broadcast
                wf.write(best_path)
        if self.is_ddp:
            torch.distributed.barrier()
        with open(tmp_file) as rf:
            best_path = rf.read()
        
        logger.info(f'Loading best_path={best_path}')
        # perform load, only for gpu training
        torch.cuda.empty_cache()
        obj = torch.load(best_path, map_location=lambda storage, loc: storage)
        model = self.trainer.get_model()
        model.load_state_dict(obj['state_dict'])
        model.cuda(self.trainer.root_gpu)
        if self.is_ddp:
            torch.distributed.barrier()
        if not (self.is_ddp and dist.get_rank() != 0):
            os.remove(tmp_file)
        torch.cuda.empty_cache()

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        "Overrides to implement warm-up for specific layers"
        if self.warmup == 1:
            one_epoch_length = len(self.trainer.train_dataloader)
            if self.trainer.global_step < one_epoch_length:
                 lr_scale = min(1., float(self.trainer.global_step + 1) / one_epoch_length)
                 for i, pg in enumerate(optimizer.param_groups):
                     after_warmup_lr = pg.get('after_warmup_lr', None)
                     if after_warmup_lr is not None:
                        pg['lr'] = lr_scale * after_warmup_lr

        super().optimizer_step(current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure)
    
    def training_step(self, batch, batch_idx):
        # TODO: try data echoing
        logits = self(batch)
        loss = self.loss(logits, batch['target'])
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss(logits, batch['target'])
        probs = torch.sigmoid(logits)
        return {'val_loss': loss, 'probs': probs.squeeze(1), 'target': batch['target'].squeeze(1)}
        
    def validation_epoch_end(self, outputs):
        def gather(t):
            gather_t = [torch.ones_like(t)] * dist.get_world_size()
            dist.all_gather(gather_t, t)
            return torch.cat(gather_t)
        avg_loss = torch.stack([out['val_loss'] for out in outputs]).mean()
        probs = torch.cat([out['probs'] for out in outputs], dim=0)
        targets = torch.cat([out['target'] for out in outputs], dim=0)
        
        if self.is_ddp:
            probs = gather(probs)
            targets = gather(targets)
            avg_loss = gather(avg_loss.unsqueeze(dim=0)).mean()
            
        auc_roc = torch.tensor(roc_auc_score(targets.detach().cpu().numpy(), probs.detach().cpu().numpy()))
        tensorboard_logs = {'val_loss': avg_loss, 'auc': auc_roc}
        
        if not (self.distributed_backend in ('ddp', 'ddp2') and dist.get_rank() != 0):
            logger.info(f'Epoch {self.current_epoch}: {avg_loss:.5f}, auc: {auc_roc:.4f}')

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        probs = torch.sigmoid(logits)
        return {'probs': probs}
    
    def test_epoch_end(self, outputs):
        probs = torch.cat([out['probs'] for out in outputs], dim=0)
        probs = probs.detach().cpu().numpy()
        self.test_predicts = probs  # Save prediction internally for easy access
        return {'the_end': 1} # We need to return something
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        arg = parser.add_argument
        arg('--model_name', default='efficientnetb4_fc', help='Name of a model for factory') # resnext50_32x4d_ssl_fc
        arg('--transform_name', type=str, default='medium_2', help='Name for transform factory')
        arg('--optimizer_name', type=str, default='sgd', help='sgd/adam/rmsprop')
        arg('--image_size', type=int, default=256, help='image size NxN')
        arg('--p', type=float, default=0.95, help='prob of an augmentation') # exp
        arg('--batch_size', type=int, default=64, help='batch_size per gpu') # 128
        arg('--lr', type=float, default=0.5)  # 1e-1
        arg('--weight_decay', type=float, default=1e5) # 5e-4
        arg('--momentum', type=float, default=0.9)
        arg('--max_epochs', type=int, default=30) # 30
        return parser


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--seed', type=int, default=666)
    arg('--distributed_backend', type=str, default='ddp')
    arg('--num_workers', type=int, default=4)
    arg('--gpus', type=int, default=2)
    arg('--num_nodes', type=int, default=1)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()
    if args.model_name is None or args.transform_name is None:
        raise ValueError('Specify model name and transformation rule')
    if args.optimizer_name not in ('sgd', 'adam', 'rmsprop'):
        raise (ValueError, 'Please choose optimizer from sgd|adam')
    #
    init_seed(seed=args.seed)
    experiment_name = md5(bytes(str(args), encoding='utf8')).hexdigest()
    logger.info(str(args)); logger.info(f'experiment_name={experiment_name}')
    tb_logger = TensorBoardLogger(save_dir=RESULT_DIR, name=experiment_name, version=int(time.time()))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=tb_logger.log_dir + "/{epoch:02d}-{auc:.4f}",
                                                   monitor='auc', mode='max', save_top_k=3, verbose=True)
    earlystop_callback = pl.callbacks.EarlyStopping(patience=6, verbose=True) # doesn't start even
    
    model = Model(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback,
                                            earlystop_callback=earlystop_callback, logger=tb_logger,
                                            # train_percent_check=0.01,
                                            # val_percent_check=0.1,
                                            )  # use_amp=False
    trainer.fit(model)
    
    # TODO: use patient ID to predict


if __name__ == '__main__':
    sys.exit(main())