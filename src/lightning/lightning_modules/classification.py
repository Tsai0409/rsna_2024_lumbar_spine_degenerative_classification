# src/lighting/lightning_modules/classification.py
# batch 在什麼時候被設定 -> 在 data_module 裡面被設定
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

# mix_images, target_a, target_b, lam = mixup(images, targets, alpha=0.5)
def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    assert alpha > 0, "alpha should be larger than 0"
    assert x.size(0) > 1, "Mixup cannot be applied to a single instance."  # 確保 batch 中至少有兩個樣本(x.size(0) > 1)； x=[batch_size, channels, height, width]

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

        # self.awp = False (all condition)
        if self.cfg.awp:  # AWP(Adversarial Weight Perturbation)（對抗性權重擾動），使模型對微小的權重變動更穩健，通常通過增加一個小的擾動來模擬最壞情況，從而提高模型的泛化能力；AWP 會在計算損失之前對模型的權重施加一個對抗性擾動，使模型在「更困難」的情況下進行訓練，以提高模型的魯棒性
            self.awp = AWP(self.model, None, adv_lr=self.cfg.adv_lr, adv_eps=self.cfg.adv_eps)

        # self.ema = False (all condition)
        if self.cfg.ema:  # EMA(Exponential Moving Average)，對模型參數建立一個指數移動平均（EMA）的版本。EMA 常用來平滑模型參數的更新，進而提高模型在驗證或測試階段的表現；EMA 是一種平滑技術，用於跟踪模型參數的移動平均值
            self.model_ema = ModelEmaV2(self.model, decay=0.998)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        images, targets = batch  # batch 是來自於 DataLoader 的資料

        # self.awp = False (all condition)
        if self.cfg.awp:
            if self.cfg.adv_start_epoch > int(self.current_epoch):  # 當前訓練 epoch 小於指定的起始 epoch（adv_start_epoch）時，調用 self.awp.perturb() 對模型權重施加對抗性擾動
                # if self.cfg.adv_attack_ratio < random.uniform(0.0, 1.0):
                #     delta = attack(self.model, images, targets, epsilon=self.cfg.adv_attack_eps)
                #     images = images + delta

                self.awp.perturb()  # Apply AWP perturbation

        # self.mixup = False (all condition)
        if self.cfg.mixup and (torch.rand(1)[0] < 0.5) and (self.cfg.warmup_epochs < self.current_epoch) and (images.size(0) > 1):  # Mixup 數據增強
            mix_images, target_a, target_b, lam = mixup(images, targets, alpha=0.5)  # mix_images：混合後的影像、target_a 與 target_b：分別是兩個來源樣本的標籤、lam：混合比例（lam ∈ [0,1]）
            # self.arcface = False (all condition)
            if self.cfg.arcface:  # 根據配置參數 arcface 的值來決定如何進行前向傳播(即計算 logits)的方式；ArcFace 機制的模型在前向計算時需要同時獲取輸入影像和對應的標籤資訊；ArcFace 是一種常用於面部識別或其它需要更嚴格區分類別的任務的技術
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
        
        else:  # here；沒有使用 Mixup 數據增強
            # self.arcface = False
            if self.cfg.arcface:
                logits = self.model(images, targets)
            else:  # here
                logits = self.forward(images)
                '''
                def forward(self, x):
                    return self.model(x)
                '''
                # logits 通常指的是神經網絡最後一層的原始輸出，即未經激活函數（如 sigmoid 或 softmax）處理的原始分數。這些分數代表了模型對不同類別的「置信度」，但是它們尚未被轉換成最終的概率
                # 無 Mixup 時：logits 是模型對原始圖像 images 進行推理的結果。
                # 使用 Mixup 時：logits 是對混合圖像 mix_images 的推理結果，並根據 target_a 和 target_b 計算損失。

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
            
            # self.criterion = torch.nn.BCEWithLogitsLoss() (all condition) -> Binary Cross-Entropy Loss(二元交叉熵損失)
            loss = self.cfg.criterion(logits, targets)  # self.criterion = torch.nn.BCEWithLogitsLoss() -> configs 中定義
            '''
            # 反向傳播和權重更新是由 PyTorch Lightning 自動處理的：
            # 這一部分的過程並不需要手動寫出，因為 Lightning 會自動在每次 training_step 結束後進行反向傳播和權重更新。這些步驟是在 trainer.fit() 內部進行的。
            # 反向傳播時：loss.backward()  # 計算梯度、 optimizer.step()  # 更新權重 -> 在 PyTorch Lightning 會自動處理這些操作
            # 進行反向傳輸的時候有什麼參數要調整嗎 ->
            #    學習率(Learning Rate）：控制每次權重更新的步伐大小，對訓練過程影響深遠。
            #    梯度裁剪(Gradient Clipping）：限制梯度的大小，防止梯度爆炸問題，保證訓練穩定性。
            #    AWP(Adversarial Weight Perturbation）：對模型權重施加微小的對抗性擾動，提升模型的魯棒性。(not use)
            #    EMA(Exponential Moving Average）：通過對模型參數進行指數移動平均，平滑訓練過程中的不穩定性。(not use)
            #    優化器(Optimizer）：自動根據計算出的梯度更新模型的權重。
            #    學習率調度器(Learning Rate Scheduler）：根據訓練進展動態調整學習率，提高訓練效果。
            #    損失函數(Loss Function）：衡量模型預測結果與實際標籤之間的誤差，用來指導權重更新。
            # 所有這些操作最終都會在 training_step 中的反向傳播過程後由 Lightning 自動處理。
            '''
        
        # self.awp = False (all condition)
        if self.cfg.awp:
            self.awp.restore()  # Restore model parameters 還原

        self.log("train_loss", loss.item(), on_step=False, on_epoch=True)
        return loss

    def on_after_backward(self):  # on_after_backward 在每次反向傳播(backward pass)之後被調用
        # self.awp = False (all condition)
        if self.cfg.awp:
            self.awp.restore()  # Restore model parameters after backward pass；在反向傳播完成後，需要將這個擾動還原，以便接下來進行的優化步驟（例如參數更新）是基於原始（未擾動）的模型參數

    def on_train_batch_end(self, outputs, batch, batch_idx):  # on_train_batch_end 在每個訓練批次結束後被調用。這個時候，一個 batch 的前向與反向傳播都已經完成，參數也已更新
        # self.ema = False (all condition)
        if self.cfg.ema:
            self.model_ema.update(self.model)

    def validation_step(self, batch, batch_nb):
        images, targets = batch
        # self.ema = False (all condition)
        if self.cfg.ema:  # 如果在配置 cfg 中啟用了 ema(Exponential Moving Average），則使用 EMA 模型進行前向傳播
            logits = self.model_ema.module(images)
        else:  # here；否則使用本身的 forward
            logits = self.forward(images)

        if isinstance(logits, tuple):  # 某些模型（尤其是多輸出或特別架構）在 forward 時會回傳 (logits, 其他資訊)。
            logits = logits[0]  # 這裡若檢測到是 tuple，僅取第一個元素作為 logits，避免後續計算出現錯誤

        loss = self.cfg.criterion(logits, targets)  # logits 表示模型對於該類別的信心分數；在深度學習中，logits 通常指的是 模型最後一層尚未經過激活函式（如 sigmoid 或 softmax）處理的原始輸出。也就是說，logits 是一個未經歸一化的分數（通常是實數值，可正可負），代表模型對各類別或輸出維度的「信心值」。
        preds = logits
        # preds = logits.sigmoid()
        output = OrderedDict( {
            "targets": targets.detach(), 
            "preds": preds.detach(), 
            "loss": loss.detach()
        })  # "targets"：存放真實標籤，即 targets.detach() 的結果、"preds"：存放模型的預測值，即 preds.detach() 的結果、"loss"：存放計算出的損失，即 loss.detach() 的結果
        return output
        '''
        # training_step 是用來處理每個訓練批次的邏輯，並且會對模型進行權重更新（如反向傳播和優化器步驟）。它還可能使用訓練時的數據增強方法（如 Mixup 和 AWP）。
        # validation_step 是用來處理每個驗證批次的邏輯，計算損失和預測結果，但不會進行權重更新。它的目的是評估模型的表現。
        '''

    # 紀錄 vaild 之後的 驗證損失(v_loss)和 評分指標(val_metric)
    def validation_epoch_end(self, outputs):
        d = dict()  # 建立字典 d：用來儲存本 epoch 的聚合結果
        d["epoch"] = int(self.current_epoch)
        d["v_loss"] = torch.stack([o["loss"] for o in outputs]).mean().item()

        targets = torch.cat([o["targets"] for o in outputs]).cpu()#.numpy()  # 將所有 targets 串接後移到 CPU
        preds = torch.cat([o["preds"] for o in outputs]).cpu()#.numpy()  # 將所有 preds 串接後移到 CPU
        
        # class rsna_sagittal_level_cl_spinal_v1、class rsna_sagittal_level_cl_nfn_v1 -> self.metric = MultiAUC(label_features=self.label_features).torch
        # class rsna_sagittal_cl -> self.metric = None
        if self.cfg.metric is None:  # 若沒有指定 metric，則用負平均損失作為評分。
            score = -d['v_loss']
        elif len(np.unique(targets)) == 1:  # 若數據不具變化（所有標籤相同），則評分設為 0。
            score = 0
        else:  # 否則，利用指定的 metric 函數計算評分。
            # st()
            score = self.cfg.metric(targets, preds)  # 計算 MultiAUC -> score 反映了模型在當前驗證批次中的預測質量

        d["val_metric"] = score
        # self.save_every_epoch_val_preds = False (all condition)
        if self.cfg.save_every_epoch_val_preds:
            np.save(f'{self.cfg.output_path}/val_preds/fold{self.cfg.fold}/epoch{self.current_epoch}.npy', preds)
        self.log_dict(d, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):  # configure_optimizers 主要影響反向傳播過程，並且只有在 訓練階段的反向傳播 中才會使用
        optimizer = get_optimizer(self.cfg)
        
        # self.awp = False (all condition)
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
    
    '''
    # src/lighting/lightning_modules/scheduler_optimizer.py
    def get_optimizer(cfg):
        # self.optimizer = 'adamw' (all condition)
        elif cfg.optimizer == 'adamw':  
            optimizer = optim.AdamW(cfg.model.parameters(), lr=cfg.lr)  # self.lr = 5.5e-5 (all condition)
        return optimizer

    def get_scheduler(cfg, optimizer):
        # self.scheduler = 'CosineAnnealingWarmRestarts' (all condition)
        elif cfg.scheduler == 'CosineAnnealingWarmRestarts':  
            scheduler = CosineAnnealingWarmRestarts(optimizer, cfg.epochs, eta_min=cfg.eta_min)  # self.eta_min = 5e-7
        return scheduler
    '''

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
