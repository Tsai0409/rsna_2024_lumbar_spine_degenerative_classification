from collections import OrderedDict
import torch.optim as optim

from timm.utils import ModelEmaV2
import pytorch_lightning as pl
import torch
import numpy as np
import random
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from pdb import set_trace as st
from .scheduler_optimizer import get_optimizer, get_scheduler

class AWP:
    def __init__(
        self, model, optimizer, *, adv_param="weight", adv_lr=0.001, adv_eps=0.001
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()
        self._attack_step()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                grad = self.optimizer.state[param]["exp_avg"]
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(
                        grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e))
                    )

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    return mixed_x, target_a, target_b, lam

def mixup_hms(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."

    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0])
    mixed_x = lam * x + (1 - lam) * x[rand_index, :]
    target_a, target_b = y, y[rand_index]
    weight_a, weight_b = w, w[rand_index]
    return mixed_x, target_a, target_b, weight_a, weight_b, lam


class MyLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super(MyLightningModule, self).__init__()
        self.model = cfg.model
        # if cfg.pretrained_path is not None:
        #     self.model.load_state_dict(torch.load(cfg.pretrained_path)['state_dict'])
        self.cfg = cfg
        if self.cfg.awp:
            self.awp = AWP(self.model, None, adv_lr=self.cfg.adv_lr, adv_eps=self.cfg.adv_eps)

        if self.cfg.ema:
            self.model_ema = ModelEmaV2(self.model, decay=0.998)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        images, targets = batch

        if self.cfg.awp:
            if self.cfg.adv_start_epoch > int(self.current_epoch):
                # if self.cfg.adv_attack_ratio < random.uniform(0.0, 1.0):
                #     delta = attack(self.model, images, targets, epsilon=self.cfg.adv_attack_eps)
                #     images = images + delta

                self.awp.perturb()  # Apply AWP perturbation

        if self.cfg.mixup and (torch.rand(1)[0] < 0.5) and (self.cfg.warmup_epochs < self.current_epoch) and (images.size(0) > 1):
            mix_images, target_a, target_b, lam = mixup(images, targets, alpha=0.5)
            if self.cfg.arcface:
                logits = self.model(mix_images, targets)
            else:
                logits = self.forward(mix_images)
                # if self.cfg.distill:
                #     with torch.no_grad():
                #         for model_n, (model, weight) in enumerate(zip(self.cfg.teacher_models, [0.2, 0.4, 0.4])):
                #             if model_n == 0:
                #                 teacher_preds = model(mix_images)*weight
                #             else:
                #                 teacher_preds += model(mix_images)*weight
            if False:
                pass
            # if self.cfg.distill:
            #     if self.cfg.distill_cancer_only:
            #         loss = self.cfg.criterion((logits[:, [0]]/self.cfg.distill_temperature).sigmoid(), (teacher_preds[:, [0]]/self.cfg.distill_temperature).sigmoid())
            #     else:
            #         loss = self.cfg.criterion((logits/self.cfg.distill_temperature).sigmoid(), (teacher_preds/self.cfg.distill_temperature).sigmoid())
            #     if self.cfg.use_origin_label:
            #         if self.cfg.criterion_for_origin_ratio == 0.5:
            #             loss2 = self.cfg.criterion_for_origin(logits[:, 1:], targets[:, 1:])
            #         else:
            #             loss2 = self.cfg.criterion_for_origin(logits, targets)
            #         loss = loss*self.cfg.criterion_for_origin_ratio + loss2*(1-self.cfg.criterion_for_origin_ratio)
            else:
                loss = self.cfg.criterion(logits, target_a) * lam + (1 - lam) * self.cfg.criterion(logits, target_b)
        else:
            if self.cfg.arcface:
                logits = self.model(images, targets)
            else:
                logits = self.forward(images)
                # if self.cfg.distill:
                #     with torch.no_grad():
                #         for model_n, (model, weight) in enumerate(zip(self.cfg.teacher_models, [0.2, 0.4, 0.4])):
                #             if model_n == 0:
                #                 teacher_preds = model(images)*weight
                #             else:
                #                 teacher_preds += model(images)*weight
            # if self.cfg.distill:
            #     if self.cfg.distill_cancer_only:
            #         loss = self.cfg.criterion((logits[:, [0]]/self.cfg.distill_temperature).sigmoid(), (teacher_preds[:, [0]]/self.cfg.distill_temperature).sigmoid())
            #     else:
            #         loss = self.cfg.criterion((logits/self.cfg.distill_temperature).sigmoid(), (teacher_preds/self.cfg.distill_temperature).sigmoid())
            #     if self.cfg.use_origin_label:
            #         if self.cfg.criterion_for_origin_ratio == 0.5:
            #             loss2 = self.cfg.criterion_for_origin(logits[:, 1:], targets[:, 1:])
            #         else:
            #             loss2 = self.cfg.criterion_for_origin(logits, targets)
            #         loss = loss*self.cfg.criterion_for_origin_ratio + loss2*(1-self.cfg.criterion_for_origin_ratio)
            # else:
            loss = self.cfg.criterion(logits, targets)
        if self.cfg.awp:
            self.awp.restore()  # Restore model parameters

        self.log("train_loss", loss.item(), on_step=False, on_epoch=True)
        return loss

    def on_after_backward(self):
        if self.cfg.awp:
            self.awp.restore()  # Restore model parameters after backward pass

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.cfg.ema:
            self.model_ema.update(self.model)

    def validation_step(self, batch, batch_nb):
        images, targets = batch
        if self.cfg.ema:
            logits = self.model_ema.module(images)
        else:
            logits = self.forward(images)

        if isinstance(logits, tuple):
            logits = logits[0]
        loss = self.cfg.criterion(logits, targets)
        preds = logits
        # preds = logits.sigmoid()
        output = OrderedDict({
            "targets": targets.detach(), "preds": preds.detach(), "loss": loss.detach()
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
        elif len(np.unique(targets)) == 1:
            score = 0
        else:
            # st()
            # score = self.cfg.metric(targets, preds)
            score = self.cfg.metric(targets, preds)


        d["val_metric"] = score
        if self.cfg.save_every_epoch_val_preds:
            np.save(f'{self.cfg.output_path}/val_preds/fold{self.cfg.fold}/epoch{self.current_epoch}.npy', preds)
        self.log_dict(d, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.cfg)
        if self.cfg.awp:
            self.awp.optimizer = optimizer  # Assign optimizer to AWP

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
