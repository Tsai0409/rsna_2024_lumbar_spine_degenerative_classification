from collections import OrderedDict
import torch.optim as optim

import pytorch_lightning as pl
import torch
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from pdb import set_trace as st
from .scheduler_optimizer import get_optimizer, get_scheduler
from torch_geometric.data import Batch
from torch_geometric.utils import degree

def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam

# def mixup(input, truth, clip=[0, 1]):
#     indices = torch.randperm(input.size(0))
#     shuffled_input = input[indices]
#     shuffled_labels = truth[indices]

#     lam = np.random.uniform(clip[0], clip[1])
#     input = input * lam + shuffled_input * (1 - lam)
#     return input, truth, shuffled_labels, lam

def decode(logits: torch.Tensor, data: Batch) -> torch.Tensor:
    sizes = degree(data.batch, dtype=torch.long).tolist()

    logit_list = logits.split(sizes)
    subset_node_idx_list = data.subset_node_idx.split(sizes)

    preds = []
    for y_pred, subset_node_idx in zip(logit_list, subset_node_idx_list):
        prob = y_pred
        # 確率が大きい上位10個を取得
        arg_idx = torch.argsort(prob, descending=True)
        arg_topk = arg_idx[:10]
        topk_node_idx = subset_node_idx[arg_topk]
        # 10個未満の場合は0を追加
        if len(topk_node_idx) < 10:
            topk_node_idx = torch.cat(
                [
                    topk_node_idx,
                    torch.zeros(
                        10 - len(topk_node_idx), dtype=torch.long, device=topk_node_idx.device
                    ),
                ]
            )

        preds.append(topk_node_idx)

    return torch.cat(preds)


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

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = self.cfg.criterion(logits, batch)

        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = self.cfg.criterion(logits, batch)
        preds = decode(logits, batch)
        self.validation_step_outputs.append(
            (
                # batch.label.detach().cpu().numpy(),
                batch.label,
                preds.detach().cpu().numpy(),
                loss.detach().cpu().numpy(),
            )
        )
        self.log(
            "v_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        return loss

    def validation_epoch_end(self, outputs):
        # st()
        labels = np.concatenate([x[0] for x in self.validation_step_outputs]).reshape(-1)
        preds = np.concatenate([x[1] for x in self.validation_step_outputs]).reshape(-1, 10)

        score = self.cfg.metric(labels.tolist(), preds.tolist(), k=self.cfg.k)
        self.log(
            "val_metric",
            score,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

        self.validation_step_outputs.clear()
        # np.save(f'{self.cfg.output_path}/val_preds/fold{self.cfg.fold}/epoch{self.current_epoch}.npy', preds)

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

    # def configure_optimizers(self):
    #     optimizer = get_optimizer(self.cfg)

    #     # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
    #     self.lr_scheduler = CosineWarmupScheduler(
    #         optimizer, warmup=2, max_iters=20
    #     )
    #     return optimizer

    # def optimizer_step(self, *args, **kwargs):
    #     super().optimizer_step(*args, **kwargs)
    #     self.lr_scheduler.step()  # Step per iteration


    # learning rate warm-up
    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
    #     # warm up lr
    #     if self.trainer.global_step < 500:
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.hparams.learning_rate

    #     # update params
    #     optimizer.step(closure=closure)

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
