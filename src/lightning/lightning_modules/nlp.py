from collections import OrderedDict

import pytorch_lightning as pl
import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from pdb import set_trace as st
from .scheduler_optimizer import get_nlp_optimizer, get_scheduler

class MyLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super(MyLightningModule, self).__init__()
        self.model = cfg.model
        # if cfg.pretrained_path is not None:
        #     self.model.load_state_dict(torch.load(cfg.pretrained_path)['state_dict'])
        self.cfg = cfg

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        if len(batch) == 5:
            inputs, masks, token_type_ids, meta, targets = batch
            outputs = self.model(inputs, masks, token_type_ids, meta)
        else:
            inputs, masks, token_type_ids, targets = batch
            outputs = self.model(inputs, masks, token_type_ids)
        loss = self.cfg.criterion(outputs, targets)
        # awp足す！
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        if len(batch) == 5:
            inputs, masks, token_type_ids, meta, targets = batch
            outputs = self.model(inputs, masks, token_type_ids, meta)
        else:
            inputs, masks, token_type_ids, targets = batch
            outputs = self.model(inputs, masks, token_type_ids)
        loss = self.cfg.criterion(outputs, targets)
        preds = outputs
        # preds = outputs.sigmoid()
        output = OrderedDict({
            "targets": targets.detach(), "preds": preds.detach(), "loss": loss.detach()
        })
        return output

    def validation_epoch_end(self, outputs):
        d = dict()
        d["epoch"] = int(self.current_epoch)
        d["v_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

        targets = torch.cat([o["targets"] for o in outputs]).cpu().numpy()
        preds = torch.cat([o["preds"] for o in outputs]).cpu().numpy()
        # score = roc_auc_score(y_true=targets, y_score=preds)
        if len(np.unique(targets)) == 1:
            score = 0
        else:
            if self.cfg.metric is None:
                score = -d["v_loss"]
            else:
                score = self.cfg.metric(targets, preds)

        d["val_metric"] = score
        self.log_dict(d, prog_bar=True)

    def configure_optimizers(self):
        optimizer = get_nlp_optimizer(self.cfg)
        # optimizer = optim.AdamW(self.cfg.model.parameters(), lr=self.cfg.lr)
        print('lr:', self.cfg.lr)
        scheduler = get_scheduler(self.cfg, optimizer)
        # import pdb;pdb.set_trace()
        if self.cfg.scheduler == 'linear_schedule_with_warmup':
            interval = 'step'
        else:
            interval = 'epoch'

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': interval,
                "monitor": 'val_metric',
                "frequency": 1
            }
        }
