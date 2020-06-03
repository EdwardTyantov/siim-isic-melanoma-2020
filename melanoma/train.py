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
        self.num_workers = self.distributed_backend == 'dp' and self.num_workers or self.num_workers // self.gpus
        
    def prepare_data(self) -> None:
        ds = Datasets(IMAGE_DIR, TRAIN_CSV, TEST_CSV, self.transform_name, self.image_size, p=0.5, val_split=0.2)
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
        else:
            parameters = self.model.parameters()

        optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)
        # TODO: add loading models
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
        
        return [optimizer], [scheduler]

    # learning rate warm-up
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        if self.warmup == 1:
            # warm up lr
            #print('current_epoch', current_epoch, 'rank=', dist.get_rank(), 'step', self.trainer.global_step, len(self.trainer.train_dataloader))
            one_epoch_length = len(self.trainer.train_dataloader)
            if self.trainer.global_step < one_epoch_length:
                 lr_scale = min(1., float(self.trainer.global_step + 1) / one_epoch_length)
                 for i, pg in enumerate(optimizer.param_groups):
                     after_warmup_lr = pg.get('after_warmup_lr', None)
                     if after_warmup_lr is not None:
                        pg['lr'] = lr_scale * after_warmup_lr

        # update params
        optimizer.step()
        optimizer.zero_grad()
    
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
        
        if self.distributed_backend in ('ddp', 'ddp2'):
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
        arg('--model_name', default='resnext50_32x4d_ssl_finetune', help='Name of a model for factory')
        arg('--transform_name', type=str, default='medium_1', help='Name for transform factory')
        arg('--image_size', type=int, default=256, help='image size NxN')
        arg('--batch_size', type=int, default=256, help='batch_size per gpu')
        arg('--lr', type=float, default=1e-4)  # 3e-4
        arg('--weight_decay', type=float, default=5e-4)
        arg('--momentum', type=float, default=0.9)
        arg('--max_epochs', type=int, default=10)
        return parser


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--seed', type=int, default=666)
    arg('--distributed_backend', type=str, default='ddp')
    arg('--num_workers', type=int, default=6)
    arg('--gpus', type=int, default=2)
    arg('--num_nodes', type=int, default=1)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()
    if args.model_name is None or args.transform_name is None:
        raise ValueError('Specify model name and transformation rule')
    #
    init_seed(seed=args.seed)
    experiment_name = md5(bytes(str(args), encoding='utf8')).hexdigest()
    logger.info(str(args)); logger.info(f'experiment_name={experiment_name}')
    tb_logger = TensorBoardLogger(save_dir=RESULT_DIR, name=experiment_name, version=int(time.time()))
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=tb_logger.log_dir + "/{epoch:02d}-{auc:.4f}",
                                                   monitor='auc', mode='max', save_top_k=3)
    
    model = Model(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback,
                                            ) #early_stop_callback=True, se_amp=False
    trainer.fit(model)
    
    # TODO: use patient ID to predict


if __name__ == '__main__':
    sys.exit(main())