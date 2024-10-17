from collections import OrderedDict
import torch.optim as optim

import pytorch_lightning as pl
import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from pdb import set_trace as st
from .scheduler_optimizer import get_optimizer, get_scheduler

def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam

class MyLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super(MyLightningModule, self).__init__()
        self.model = cfg.model
        # if cfg.pretrained_path is not None:
        #     self.model.load_state_dict(torch.load(cfg.pretrained_path)['state_dict'])
        self.cfg = cfg
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        images, targets = batch
        if self.cfg.mixup and (torch.rand(1)[0] < 0.5) and (self.cfg.warmup_epochs < self.current_epoch):
            mix_images, target_a, target_b, lam = mixup(images, targets, alpha=0.5)
            logits = self.forward(mix_images)
            loss = self.cfg.criterion(logits, target_a) * lam + (1 - lam) * self.cfg.criterion(logits, target_b)
        else:
            logits = self.forward(images)
            loss = self.cfg.criterion(logits, targets)
        self.log("train_loss", loss.item(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_nb):
        images, targets = batch
        logits = self.forward(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = self.cfg.criterion(logits, targets)
        preds = logits
        output = OrderedDict({
            "targets": targets.detach(), "preds": preds.detach(), "loss": loss.detach()
        })
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        d = dict()
        d["epoch"] = int(self.current_epoch)

        # targets = torch.cat([o["targets"] for o in outputs]).float()
        # preds = torch.cat([o["preds"] for o in outputs]).float()

        targets = torch.cat([o["targets"] for o in outputs])#.numpy()
        preds = torch.cat([o["preds"] for o in outputs])#.numpy()

        loss = self.cfg.criterion(preds, targets)
        d["v_loss"] = loss.item()
        if self.cfg.metric is None:
            score = -d['v_loss']
        elif len(np.unique(targets)) == 1:
            score = 0
        else:
            score = self.cfg.metric(targets.cpu(), preds.cpu())

        d["val_metric"] = score
        print('val metric:', score)

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

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if (hasattr(self.cfg, 'weight_limit')) and (self.cfg.weight_limit):
            for param in self.parameters():
                param.data.clamp_(self.cfg.weight_limit[0], self.cfg.weight_limit[1])
