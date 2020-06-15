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
from utils import init_seed, accuracy
from models import factory as model_factory
from models.losses import FocalLoss
from models.gradients import GradientEstimator
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
        self.has_two_heads = self.model_name.find('2head') != -1 and self.diagnosis_loss_weight and \
                             self.diagnosis_loss_weight > 0
        
    def prepare_data(self) -> None:
        ds = Datasets(IMAGE_DIR, TRAIN_CSV, TEST_CSV, self.transform_name, self.image_size, p=self.p, val_split=0.2)
        self.train_dataset = ds.train_dataset
        self.val_dataset = ds.val_dataset
        self.test_dataset = ds.test_dataset
        
    def log(self, *args, **kwargs):
        if not (self.is_ddp and dist.get_rank() != 0):
            logger.info(*args, **kwargs)
            
    def forward(self, batch):
        # TODO: add other features
        out = self.model(batch['image'])
        if self.has_two_heads:
            assert isinstance(out, tuple)
            return {'target': out[0], 'diagnosis': out[1]}
        
        return {'target': out}
    
    def init_loss(self):
        self.loss = FocalLoss(ce_func='bce')
        self.d_loss = FocalLoss(ce_func='ce')

    def train_dataloader(self):
        # TODO: add sampler, great imbalance of zeros, don't forget to replace_sampler_ddp ?
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)
    
    def configure_optimizers(self):
        self.warmup = 0
        if self.layers:
            parameters = []
            for param in self.model.parameters():
                param.requires_grad = False
            
            for layer, scale in self.layers['trained']:
                self.warmup = 1
                lr = scale * self.lr
                parameters.append({'params': layer.parameters(), 'lr': lr, 'after_warmup_lr': lr})
                for param in layer.parameters():
                    param.requires_grad = True
                    
            for layer, scale in self.layers['untrained']:
                parameters.append({'params': layer.parameters(), 'lr': scale*self.lr})
                for param in layer.parameters():
                    param.requires_grad = True
            self.log(str(parameters))
        else:
            parameters = self.model.parameters()

        if self.optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'sgd':
            self.optimizer = torch.optim.SGD(parameters, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.optimizer_name == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(parameters, lr=self.lr, weight_decay=self.weight_decay)

        self._grad_estimator = GradientEstimator(self.model, parameters)
        
        scheduler = ReduceLROnPlateau(self.optimizer, patience=2, factor=0.2, verbose=True, threshold_mode='abs',
                                      threshold=1e-6, callback=self.load_best_model)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        
        return [self.optimizer], [scheduler]

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

        # perform load, only for gpu training
        self.log(f'Loading best_path={best_path}')
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
        batch_logits = self(batch)
        logits = batch_logits['target']
        t_loss = self.loss(logits, batch['target'])
        weighted_losses = {'target_loss': t_loss}
        
        if self.has_two_heads:
            d_logits = batch_logits['diagnosis']
            d_targets = batch['diagnosis']
            index = d_targets.data != (-1)  # select only valid diagnosis
            # weights = self.train_dataset.diagnosis_weights.to(d_logits.device)
            d_loss = self.d_loss(d_logits[index], d_targets[index]) if d_logits[index].size(0) > 0 else 0
            weighted_losses['diagnosis_loss'] = d_loss * self.diagnosis_loss_weight
        
        loss = sum(weighted_losses.values())
        metrics = {'train_loss': loss}
        if batch_idx and self.log_steps_grad and batch_idx % self.log_steps_grad == 0:
            grad_metrics = self._grad_estimator(weighted_losses)
            metrics.update(grad_metrics)
        metrics.update(weighted_losses)
        
        return {'loss': loss, 'log': metrics}

    def validation_step(self, batch, batch_idx):
        batch_logits = self(batch)
        logits = batch_logits['target']
        loss = self.loss(logits, batch['target'])
        probs = torch.sigmoid(logits)
        metrics = {'val_loss': loss, 'probs': probs.squeeze(1), 'target': batch['target'].squeeze(1)}
        
        if self.has_two_heads:
            d_logits = batch_logits['diagnosis']
            d_targets = batch['diagnosis']
            index = d_targets.data != (-1)
            metrics['d_acc'] = accuracy(d_logits[index], d_targets[index])[0]

        return metrics
        
    def validation_epoch_end(self, metric_list):
        def gather(t):
            gather_t = [torch.ones_like(t)] * dist.get_world_size()
            dist.all_gather(gather_t, t)
            return torch.cat(gather_t)
        probs = torch.cat([out['probs'] for out in metric_list], dim=0)
        targets = torch.cat([out['target'] for out in metric_list], dim=0)
        avg_loss = torch.stack([out['val_loss'] for out in metric_list]).mean()
        if self.has_two_heads:
            d_accs = torch.stack([out['d_acc'] for out in metric_list]).mean()
        
        if self.is_ddp:
            probs = gather(probs)
            targets = gather(targets)
            avg_loss = gather(avg_loss.unsqueeze(dim=0)).mean()
            avg_d_acc = gather(d_accs.unsqueeze(dim=0)).mean() if self.has_two_heads else float('nan')
            
        auc_roc = torch.tensor(roc_auc_score(targets.detach().cpu().numpy(), probs.detach().cpu().numpy()))
        # that's used for all callbacks, why?!, val loss for reduce on plateau
        tensorboard_logs = {'val_loss': avg_loss, 'auc': auc_roc, 'd_acc': avg_d_acc,
                            'lr': self.optimizer.param_groups[0]['lr']}
        self.log(f'Epoch {self.current_epoch}: {avg_loss:.5f}, auc: {auc_roc:.5f}, d_acc: {avg_d_acc:.5f}')

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        batch_logits = self(batch)
        probs = torch.sigmoid(batch_logits['target'])
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
        arg('--model_name', default='resnext50_32x4d_ssl_fc', help='Name of a model for factory')  # resnext50_32x4d_ssl_fc
        arg('--transform_name', type=str, default='medium_2', help='Name for transform factory')
        arg('--optimizer_name', type=str, default='sgd', help='sgd/adam/rmsprop')
        arg('--image_size', type=int, default=256, help='image size NxN')
        arg('--p', type=float, default=0.95, help='prob of an augmentation')  # exp
        arg('--batch_size', type=int, default=128, help='batch_size per gpu')  # 128
        arg('--lr', type=float, default=0.1)  # 0.1
        arg('--weight_decay', type=float, default=5e-4)  # 5e-4
        arg('--momentum', type=float, default=0.9)
        arg('--max_epochs', type=int, default=30)
        arg('--use_amp', type=bool, default=True)
        # secondary params
        arg('--diagnosis_loss_weight', type=float, default=None, help='abs weight for the secondary head')  # 0.1
        arg('--log_steps_grad', type=int, default=None, help='How often to store gradients from losses to TensorBoard') # TODO: affects loop check for low LR
        return parser


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--seed', type=int, default=666)
    arg('--distributed_backend', type=str, default='ddp')
    arg('--num_workers', type=int, default=16)
    arg('--gpus', type=int, default=8)
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
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=tb_logger.log_dir + "/{epoch:02d}-{val_loss:.5f}-{auc:.5f}",
                                                   monitor='val_loss', mode='min', save_top_k=5, verbose=True)
    # set min_delta negative, b/c there's double run of early-stopping sometimes for v0.7.6
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='auc', mode='max', patience=11, #min_delta=-0.000001,
                                                     verbose=True)
    
    model = Model(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback,
                                            early_stop_callback=early_stop_callback, logger=tb_logger,
                                            # train_percent_check=0.1,
                                            # num_sanity_val_steps=args.gpus,
                                            )
    trainer.fit(model)
    
    # TODO: use patient ID to predict


if __name__ == '__main__':
    sys.exit(main())