from collections import OrderedDict
import torch.optim as optim

import pytorch_lightning as pl
import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from pdb import set_trace as st
from .scheduler_optimizer import get_optimizer, get_scheduler

class MyLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super(MyLightningModule, self).__init__()
        self.model = cfg.model
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        ids = batch['ids'].to(torch.long)
        mask = batch['mask'].to(torch.long)
        num_vals = batch['num_vals'].to(torch.float)
        labels = batch['label'].to(torch.long)
        if self.cfg.arcface:
            logits = self.model(ids, mask, num_vals, labels)
        else:
            logits = self.model(ids, mask, num_vals)
        loss = self.cfg.criterion(logits, labels)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        ids = batch['ids'].to(torch.long)
        mask = batch['mask'].to(torch.long)
        num_vals = batch['num_vals'].to(torch.float)
        labels = batch['label'].to(torch.long)
        logits = self.model(ids, mask, num_vals)
        loss = self.cfg.criterion(logits, labels)
        output = OrderedDict({
            "targets": labels.detach(), "preds": logits.detach(), "loss": loss.detach()
        })
        return output

    def validation_epoch_end(self, outputs):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        d["v_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

        targets = torch.cat([o["targets"] for o in outputs]).cpu()#.numpy()
        preds = torch.cat([o["preds"] for o in outputs]).cpu()#.numpy()
        if self.cfg.metric is None:
            score = -d['v_loss']
        # elif len(np.unique(targets)) == 1:
        #     score = 0
        else:
            if len(np.unique(targets))==1:
                return 0
            score = self.cfg.metric(targets, preds)

        d["val_metric"] = score
        self.log_dict(d, prog_bar=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_scheduler(self.cfg, optimizer),
                "monitor": 'val_metric',
                "frequency": 1
            }
        }
